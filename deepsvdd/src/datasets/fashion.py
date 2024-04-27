from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import FashionMNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, ApplyGlobalContrastNormalization, CheckIfOutlier

import torchvision.transforms as transforms


class FASHION_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-2.681241989135742, 24.854305267333984),
                   (-2.57785701751709, 11.16978931427002),
                   (-2.8081703186035156, 19.133543014526367),
                   (-1.9533653259277344, 18.656726837158203),
                   (-2.6103854179382324, 19.166683197021484),
                   (-1.2358521223068237, 28.46310806274414),
                   (-3.251605987548828, 24.19683265686035),
                   (-1.0814441442489624, 16.87881851196289),
                   (-3.656097888946533, 11.350274085998535),
                   (-1.3859288692474365, 11.426652908325195)]

        # Instantiating transformation classes
        gcn_transform = ApplyGlobalContrastNormalization('l1')
        outlier_transform = CheckIfOutlier(self.outlier_classes)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(gcn_transform),
            transforms.Normalize(
                [min_max[normal_class][0]],  # Normalization parameters
                [min_max[normal_class][1] - min_max[normal_class][0]]
            )
        ])

        target_transform = transforms.Lambda(outlier_transform)

        train_set = MyFASHION(root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyFASHION(root=self.root, train=False, download=True,
                                  transform=transform, target_transform=target_transform)


class MyFASHION(FashionMNIST):
    """Torchvision FashionMNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyFASHION, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the FashionMNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
