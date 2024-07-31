import torch
import numpy as np
import logging
import torchvision.transforms as transforms
import torch.nn.functional as F

# Pre-computed min and max values (after applying GCN) from train data per class
MIN_MAX = [(-0.8826567065619495, 9.001545489292527),
            (-0.6661464580883915, 20.108062262467364),
            (-0.7820454743183202, 11.665100841080346),
            (-0.7645772083211267, 12.895051191467457),
            (-0.7253923114302238, 12.683235701611533),
            (-0.7698501867861425, 13.103278415430502),
            (-0.778418217980696, 10.457837397569108),
            (-0.7129780970522351, 12.057777597673047),
            (-0.8280402650205075, 10.581538445782988),
            (-0.7369959242164307, 10.697039838804978)]
        
def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x


class ApplyGlobalContrastNormalization:
    def __init__(self, scale='l1'):
        self.scale = scale

    def __call__(self, x):
        return global_contrast_normalization(x, self.scale)


def feature_encoder_transform(x: torch.Tensor, digit: int = 3):
     
    transform = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Lambda(ApplyGlobalContrastNormalization('l1')),
        transforms.Normalize(
            [MIN_MAX[digit - 1][0]],  # Normalization parameters
            [MIN_MAX[digit - 1][1] - MIN_MAX[digit - 1][0]]
        )
    ])
    
    return transform(x)

# TODO: probably we won't need this FeatureEncoder anymore or 
# can be refactored to accept the pretrained encoder, pretrained decoder and the mean net (fc1) 
class FeatureEncoder(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__) 

        self.rep_dim = 32
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = torch.nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = torch.nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = torch.nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = torch.nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = torch.nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = torch.nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = torch.nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x
    
    def reconstruct(self, x):
         
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        
        return x
    
    
class PretrainedEncoder(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.rep_dim = 32
        self.pool = torch.nn.MaxPool2d(2, 2)
 
        self.conv1 = torch.nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = torch.nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(4, eps=1e-04, affine=False)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
         
        return x
        
class PretrainedDecoder(torch.nn.Module):
       def __init__(self):
        super().__init__() 
        self.deconv1 = torch.nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = torch.nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = torch.nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = torch.nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)
        
        def forward(self, x):
         
            x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
            x = F.interpolate(F.leaky_relu(x), scale_factor=2)
            x = self.deconv1(x)
            x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
            x = self.deconv2(x)
            x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
            x = self.deconv3(x)
            x = torch.sigmoid(x)
        
            return x
    
class MeanNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.rep_dim = 32
        self.fc1 = torch.nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)
        
    def forward(self, x):
        x = self.fc1(x)
        
        return x
