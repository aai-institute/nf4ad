from typing import Iterable, Optional, Type, Any, Dict, List
import torch
from pyro import distributions as dist

# Reuse base Flow implementation from USFlows
from src.usflows.flows import Flow as _BaseFlow

# reuse auxiliary transforms from USFlows
from src.usflows.transforms import (
    ScaleTransform,
    LUTransform,
    InverseTransform,
    BaseTransform,
    BlockAffineTransform,
    HouseholderTransform,
    SequentialAffineTransform,
)

# Use our MaskedAffineCoupling instead of MaskedCoupling
from nf4ad.transforms import MaskedAffineCoupling


# Export Flow name so other nf4ad modules importing .flows.Flow continue to work
Flow = _BaseFlow


class NonUSFlow(_BaseFlow):
    """Flow identical to USFlow but using MaskedAffineCoupling (affine coupling) instead of MaskedCoupling (additive).

    Signature and behavior mirror src.usflows.flows.USFlow.
    """

    MASKTYPE = ("checkerboard", "channel")

    def __init__(
        self,
        base_distribution: dist.Distribution,
        in_dims: List[int],
        coupling_blocks: int,
        conditioner_cls: Type[torch.nn.Module],
        conditioner_args: Dict[str, Any],
        soft_training: bool = False,
        prior_scale: Optional[float] = None,
        training_noise_prior=None,
        affine_conjugation: bool = False,
        nonlinearity: Optional[torch.nn.Module] = None,
        lu_transform: int = 1,
        householder: int = 1,
        masktype: str = "checkerboard",
        device: str = "cpu",
        *args,
        **kwargs
    ):
        self.coupling_blocks = coupling_blocks
        self.in_dims = in_dims
        self.soft_training = soft_training
        self.training_noise_prior = training_noise_prior
        self.conditioner_cls = conditioner_cls
        self.conditioner_args = conditioner_args
        self.prior_scale = prior_scale
        self.device = device

        if masktype == "checkerboard":
            self.mask_Generator = NonUSFlow.create_checkerboard_mask
        elif masktype == "channel":
            self.mask_Generator = NonUSFlow.create_channel_mask
        else:
            raise ValueError(f"Unknown mask type {masktype}")

        if lu_transform < 0:
            raise ValueError("Number of LU transforms must be non-negative")
        self.lu_transform = lu_transform

        if householder < 0:
            raise ValueError("Number of Householder vectors transforms must be non-negative")
        self.householder = householder

        mask = self.mask_Generator(in_dims)
        layers = []

        for i in range(coupling_blocks):
            affine_layers = []
            # LU layers
            for _ in range(lu_transform):
                lu_layer = LUTransform(in_dims[0], prior_scale)
                affine_layers.append(lu_layer)

            # Householder
            if householder > 0:
                householder_layer = HouseholderTransform(dim=in_dims[0], nvs=householder, device=self.device)
                affine_layers.append(householder_layer)

            block_affine_layer = None
            if len(affine_layers) > 0:
                block_affine_layer = BlockAffineTransform(in_dims, SequentialAffineTransform(affine_layers))
                layers.append(block_affine_layer)

            # Here: use MaskedAffineCoupling instead of additive MaskedCoupling
            coupling_layer = MaskedAffineCoupling(mask, conditioner_cls(**conditioner_args))
            layers.append(coupling_layer)

            # Inverse affine transform if requested
            if affine_conjugation and block_affine_layer is not None:
                layers.append(InverseTransform(block_affine_layer))

            # alternate mask
            mask = 1 - mask

        # final LU + scale as in original USFlow
        lu_layer = LUTransform(in_dims[0], prior_scale)
        block_affine_layer = BlockAffineTransform(in_dims, lu_layer)
        layers.append(block_affine_layer)
        scale_layer = ScaleTransform(in_dims)
        layers.append(scale_layer)

        # Initialize base Flow with constructed layers
        super().__init__(
            base_distribution,
            layers,
            soft_training=soft_training,
            training_noise_prior=training_noise_prior,
            device=device,
            *args,
            **kwargs,
        )

    @classmethod
    def create_checkerboard_mask(cls, in_dims, invert: bool = False) -> torch.Tensor:
        axes = [torch.arange(d, dtype=torch.int32) for d in in_dims]
        ax_idxs = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        mask = torch.fmod(ax_idxs.sum(dim=0), 2)
        mask = mask.to(torch.float32).view(1, *in_dims)
        if invert:
            mask = 1 - mask
        return mask

    @classmethod
    def create_channel_mask(cls, in_dims, invert: bool = False) -> torch.Tensor:
        axes = [torch.arange(d, dtype=torch.int32) for d in in_dims]
        ax_idxs = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        mask = torch.fmod(ax_idxs[0], 2)
        mask = mask.to(torch.float32).view(1, *in_dims)
        if invert:
            mask = 1 - mask
        return mask

    def log_prior(self) -> torch.Tensor:
        if self.prior_scale is not None:
            log_prior = 0
            for p in self.layers:
                try:
                    log_prior = log_prior + p.log_prior()
                except Exception:
                    # some layers may not implement log_prior
                    continue
            return log_prior
        else:
            return 0

    # other methods (log_prob, sample, to, simplify) are inherited from _BaseFlow and behave the same