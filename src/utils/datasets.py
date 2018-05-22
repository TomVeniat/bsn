import logging
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST, CIFAR10, SVHN, CIFAR100

logger = logging.getLogger(__name__)


def get_CIFAR10(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 32
    in_channels = 3
    out_size = (10,)
    val_size = 5000

    data_augmentation = [transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip()]

    normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transfrom = transforms.Compose([
        transforms.Compose(data_augmentation),
        transforms.ToTensor(),
        normalization])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalization])

    train_set = CIFAR10(root=path, train=True, download=True, transform=train_transfrom)
    test_set = CIFAR10(root=path, train=False, download=True, transform=test_transform)
    train_set, val_set = validation_split(train_set, train_transfrom, test_transform, val_size=val_size)

    return train_set, val_set, test_set, img_dim, in_channels, out_size


def get_CIFAR100(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 32
    in_channels = 3
    out_size = (100,)
    val_size = 5000

    data_augmentation = [transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip()]

    normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transfrom = transforms.Compose([
        transforms.Compose(data_augmentation),
        transforms.ToTensor(),
        normalization])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalization])

    train_set = CIFAR100(root=path, train=True, download=True, transform=train_transfrom)
    test_set = CIFAR100(root=path, train=False, download=True, transform=test_transform)
    train_set, val_set = validation_split(train_set, train_transfrom, test_transform, val_size=val_size)

    return train_set, val_set, test_set, img_dim, in_channels, out_size


def get_SVHN(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 32
    in_channels = 3
    out_size = (10,)
    val_size = 7500

    data_augmentation = [transforms.Pad(4),
                         transforms.RandomCrop(32)]

    normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transfrom = transforms.Compose([
        transforms.Compose(data_augmentation),
        transforms.ToTensor(),
        normalization])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalization])

    train_set = SVHN(root=path, split='train', download=True, transform=train_transfrom)
    test_set = SVHN(root=path, split='test', download=True, transform=test_transform)
    train_set, val_set = validation_split(train_set, train_transfrom, test_transform, val_size=val_size)

    return train_set, val_set, test_set, img_dim, in_channels, out_size


def get_MNIST(path, *args):
    img_dim = 32
    in_channels = 1
    out_size = (10,)
    val_size = 10000

    data_augmentation = [transforms.Pad(4),
                         transforms.RandomCrop(32)]

    normalization = transforms.Normalize((0.1307,), (0.3081,))

    train_transfrom = transforms.Compose([
        transforms.Compose(data_augmentation),
        transforms.ToTensor(),
        normalization]
    )

    test_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        normalization]
    )

    train_set = MNIST(root=path, train=True, download=True, transform=train_transfrom)
    test_set = MNIST(root=path, train=False, download=True, transform=test_transform)
    train_set, val_set = validation_split(train_set, train_transfrom, test_transform, val_size=val_size)

    return train_set, val_set, test_set, img_dim, in_channels, out_size


def get_PartLabels(path, *args):
    from torchdata.datasets.PartLabels import PartLabels
    from torchdata.transforms import jointransforms
    img_dim = 256
    in_channels = 3
    out_size = (3, 256, 256)

    train_transform = [jointransforms.JointPad(6), jointransforms.JointRandomCrop(img_dim),
                       jointransforms.JointRandomHorizontalFlip()]
    train_transform = jointransforms.JointCompose(train_transform)

    test_transform = jointransforms.JointPad(3)

    input_transfroms = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    input_transfroms = transforms.Compose(input_transfroms)

    def label_transform(x):
        x[1] *= 2
        x[2] *= 0
        return x.sum(0)

    out_transfroms = [transforms.ToTensor(),
                      transforms.Lambda(lambda x: label_transform(x))]
    out_transfroms = transforms.Compose(out_transfroms)

    train_set = PartLabels(root=path, train=True, download=True, transform=input_transfroms,
                           target_transform=out_transfroms, joint_transform=train_transform)
    val_set = PartLabels(root=path, validation=True, download=True, transform=input_transfroms,
                         target_transform=out_transfroms, joint_transform=test_transform)
    test_set = PartLabels(root=path, test=True, download=True, transform=input_transfroms,
                          target_transform=out_transfroms, joint_transform=test_transform)

    return train_set, val_set, test_set, img_dim, in_channels, out_size


def get_ImageNet(path, *args):
    img_dim = 256
    in_channels = 3
    out_size = (1000,)

    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_set = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.Pad(16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    val_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.Pad(16),
        transforms.ToTensor(),
        normalize,
    ]))

    return train_set, val_set, None, img_dim, in_channels, out_size


sets = {
    'CIFAR10': get_CIFAR10,
    'CIFAR100': get_CIFAR100,
    'MNIST': get_MNIST,
    'SVHN': get_SVHN,
    'PART': get_PartLabels,
    'ImageNet': get_ImageNet,
}


def get_data(ds_name, batch_size, path, args=None):
    logger.debug("Using {} dataset".format(ds_name))

    if ds_name in sets.keys():
        train_set, val_set, test_set, img_dim, in_channels, out_size = sets[ds_name](path, args)
    else:
        raise ValueError("Dataset must in {}, got {}".format(sets.keys(), ds_name))

    logger.debug("N train : %d" % len(train_set))
    logger.debug("N valid : %d" % len(val_set))
    logger.debug("N test : %d" % len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4) if train_set is not None else None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4) if test_set is not None else None
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4) if val_set is not None else None

    data_properties = {
        'img_dim': img_dim,
        'in_channels': in_channels,
        'out_size': out_size
    }

    return train_loader, val_loader, test_loader, data_properties


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length, transform):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        self.transform = transform
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        self.parent_ds.transform = self.transform
        return self.parent_ds[i + self.offset]


def validation_split(dataset, train_transforms, val_transforms, val_size=None, val_share=0.1):
    """
       Split a (training and validation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

       """

    val_offset = len(dataset) - val_size if val_size is not None else int(len(dataset) * (1 - val_share))
    assert val_offset > 0, "Can't extract a size {} validation set out of a size {} dataset".format(val_size,
                                                                                                    len(dataset))
    return PartialDataset(dataset, 0, val_offset, train_transforms), PartialDataset(dataset, val_offset,
                                                                                    len(dataset) - val_offset,
                                                                                    val_transforms)
