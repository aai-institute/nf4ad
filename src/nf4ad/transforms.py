import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.usflows.transforms import BaseTransform
from pyro import distributions as dist

class MaskedAffineCoupling(BaseTransform):
	"""
	Masked affine coupling layer that preserves input/event shape (no flattening).
	The conditioner is expected to accept the masked input (same shape as x) and
	either return:
	  - a tuple/list (s, t) both with same shape as x, or
	  - a single tensor with channels == 2 * x.channels (split into s,t along channel dim), or
	  - a single tensor with same shape as x (interpreted as additive shift t; s=0).
	Args:
		mask (torch.Tensor): binary mask with same event shape as x (no batch dim).
		conditioner (nn.Module): mapping masked inputs -> params as described above.
		scale_activation (str): 'exp' (default) or 'softplus' for positive scale mapping.
		clamp (float): maximum absolute value for tanh before exponentiating to avoid extreme scales.
	"""
	bijective = True

	def __init__(
		self,
		mask: torch.Tensor,
		conditioner: nn.Module,
		scale_activation: str = "exp",
		clamp: float = 5.0,
	):
		super().__init__()
		self.register_buffer("mask", mask.float())
		self.conditioner = conditioner
		self.scale_activation = scale_activation
		self.clamp = float(clamp)

		self.domain = dist.constraints.real_vector
		self.codomain = dist.constraints.real_vector

	def _parse_params(self, params, x):
		"""Return (s, t) each same shape as x given various conditioner outputs."""
		if isinstance(params, (list, tuple)) and len(params) == 2:
			s, t = params
		else:
			# params is a single tensor
			if params.shape == x.shape:
				# additive-only: use zero s
				s = torch.zeros_like(params)
				t = params
			else:
				# try splitting along channel dimension if possible
				if params.dim() >= 2 and x.dim() >= 2 and params.shape[1] == 2 * x.shape[1]:
					c = x.shape[1]
					s = params[:, :c, ...]
					t = params[:, c:, ...]
				else:
					raise ValueError(
						"Conditioner output shape not compatible. "
						"Expected (s,t) tuple, tensor same shape as x, or tensor with 2*C channels."
					)
		# ensure same device/dtype as x
		s = s.to(x.dtype)
		t = t.to(x.dtype)
		return s, t

	def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""
		Forward: y = mask * x + (1-mask) * (scale * x + shift)
		"""
		x_masked = x * self.mask
		if context is None:
			params = self.conditioner(x_masked)
		else:
			params = self.conditioner(x_masked, context)

		s, t = self._parse_params(params, x)

		# stable log-scale
		log_s = torch.tanh(s) * self.clamp
		if self.scale_activation == "exp":
			scale = torch.exp(log_s)
			log_scale = log_s
		elif self.scale_activation == "softplus":
			scale = F.softplus(log_s) + 1e-6
			log_scale = torch.log(scale + 1e-12)
		else:
			raise ValueError("Unsupported scale_activation")

		y = x_masked + (1.0 - self.mask) * (x * scale + t)
		return y

	def backward(self, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""
		Inverse: x = mask * y + (1-mask) * ((y - shift) / scale)
		"""
		y_masked = y * self.mask
		if context is None:
			params = self.conditioner(y_masked)
		else:
			params = self.conditioner(y_masked, context)

		s, t = self._parse_params(params, y)

		log_s = torch.tanh(s) * self.clamp
		if self.scale_activation == "exp":
			scale = torch.exp(log_s)
		elif self.scale_activation == "softplus":
			scale = F.softplus(log_s) + 1e-6
		else:
			raise ValueError("Unsupported scale_activation")

		x = y_masked + (1.0 - self.mask) * ((y - t) / (scale + 1e-12))
		return x

	def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""
		Log absolute determinant: sum over unmasked dims of log|scale| per batch element.
		Returns tensor shape (batch,)
		"""
		# obtain s via conditioner on masked x (as in forward)
		x_masked = x * self.mask
		if context is None:
			params = self.conditioner(x_masked)
		else:
			params = self.conditioner(x_masked, context)
		s, _ = self._parse_params(params, x)

		log_s = torch.tanh(s) * self.clamp
		if self.scale_activation == "exp":
			log_scale = log_s
		elif self.scale_activation == "softplus":
			log_scale = torch.log(F.softplus(log_s) + 1e-12)
		else:
			raise ValueError("Unsupported scale_activation")

		# Only unmasked positions contribute
		contrib = (1.0 - self.mask) * log_scale
		# sum over event dims (all dims except batch)
		# produce per-batch sums
		return contrib.view(contrib.shape[0], -1).sum(dim=1)

	def is_feasible(self) -> bool:
		# feasible if mask contains only 0/1
		m = self.mask
		return ((m == 0) | (m == 1)).all()

	def jitter(self, jitter: float = 1e-6) -> None:
		# no-op for now
		return None