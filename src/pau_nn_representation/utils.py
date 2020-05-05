from torchvision.datasets import CIFAR10


class CIFAR10_representations(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_representations, self).__init__(root, train=train, transform=transform,
                                  target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = super.__getitem__(index)
        return img, target, index