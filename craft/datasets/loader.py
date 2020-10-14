import nonechucks as nc
from .. import transforms as TC
from .custom import CustomDataset
from .synthtext import SynthTextDataset
from torch.utils.data.dataloader import DataLoader




mean_norm = [0.485, 0.456, 0.406]
std_norm = [0.229, 0.224, 0.225]


def trainset_transform(resize_size=(224, 224), img_size=(224, 224), randrot_deg=15, scale=0.5,):
    transform = TC.Compose([
        TC.Resize(size=resize_size),
        TC.RandomRotation(degrees=randrot_deg),
        TC.RandomCrop(size=img_size),
        # TC.RandomVerticalFlip(),
        TC.RandomHorizontalFLip(),
        TC.RandomContrast(),
        TC.RandomBrightness(),
        TC.RandomColor(),
        TC.RandomSharpness(),
        # TC.RandomHue(),
        TC.ScaleRegionAffinity(scale=scale),
        TC.NumpyToTensor(),
        TC.Normalize(mean=mean_norm, std=std_norm)
    ])

    return transform


def validset_transform(resize_size=(224, 224), scale=0.5):
    transform = TC.Compose([
        TC.Resize(size=(224, 224)),
        TC.ScaleRegionAffinity(scale=0.5),
        TC.NumpyToTensor(),
        TC.Normalize(mean=mean_norm, std=std_norm)
    ])

    return transform


def _custom_loader(path, mode='train', batch_size=16, shuffle=True, nworkers=4,
                   imsize=(224, 224), randrot=15, scale=0.5,
                   gksize=(35, 35), gdratio=2.0, guse_pad=False,
                   gpad_factor=0.1, aff_thresh=0.1):

    if mode == 'train':
        tfrm = trainset_transform(img_size=imsize, resize_size=imsize,
                                  randrot_deg=randrot, scale=scale)
    else:
        tfrm = validset_transform(resize_size=imsize, scale=scale)

    dset = CustomDataset(path, mode, transform=tfrm, aff_thresh=aff_thresh,
                         gauss_ksize=gksize, gauss_dratio=gdratio,
                         gauss_use_pad=guse_pad, gauss_pad_factor=gpad_factor)

    dset = nc.SafeDataset(dset)

    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=nworkers, drop_last=True)

    return loader


def custom_trainloader(path, batch_size=16, shuffle=True, nworkers=4,
                       imsize=(224, 224), randrot=15, scale=0.5,
                       gksize=(35, 35), gdratio=2.0, guse_pad=False,
                       gpad_factor=0.1, aff_thresh=0.1):

    return _custom_loader(path, mode='train', batch_size=batch_size, shuffle=shuffle,
                          nworkers=nworkers, imsize=imsize, randrot=randrot, scale=scale,
                          gksize=gksize, gdratio=gdratio, guse_pad=guse_pad,
                          gpad_factor=gpad_factor, aff_thresh=aff_thresh)


def custom_validloader(path, batch_size=16, shuffle=True, nworkers=4,
                       imsize=(224, 224), randrot=15, scale=0.5,
                       gksize=(35, 35), gdratio=2.0, guse_pad=False,
                       gpad_factor=0.1, aff_thresh=0.1):

    return _custom_loader(path, mode='valid', batch_size=batch_size, shuffle=shuffle,
                          nworkers=nworkers, imsize=imsize, randrot=randrot, scale=scale,
                          gksize=gksize, gdratio=gdratio, guse_pad=guse_pad,
                          gpad_factor=gpad_factor, aff_thresh=aff_thresh)


def _synthtext_loader(path, mode='train', batch_size=16, shuffle=True, nworkers=4,
                      imsize=(224, 224), randrot=15, scale=0.5,
                      gksize=(35, 35), gdratio=2.0, guse_pad=False,
                      gpad_factor=0.1, aff_thresh=0.1):

    if mode == 'train':
        tfrm = trainset_transform(img_size=imsize, resize_size=imsize,
                                  randrot_deg=randrot, scale=scale)
    else:
        tfrm = validset_transform(resize_size=imsize, scale=scale)

    dset = SynthTextDataset(root=path, mode=mode, transform=tfrm, aff_thresh=aff_thresh,
                            gauss_ksize=gksize, gauss_dratio=gdratio,
                            gauss_use_pad=guse_pad, gauss_pad_factor=gpad_factor)

    dset = nc.SafeDataset(dset)

    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=nworkers, drop_last=True)

    return loader


def synthtext_trainloader(path, batch_size=16, shuffle=True, nworkers=4,
                          imsize=(224, 224), randrot=15, scale=0.5,
                          gksize=(35, 35), gdratio=2.0, guse_pad=False,
                          gpad_factor=0.1, aff_thresh=0.1):

    return _synthtext_loader(path, mode='train', batch_size=batch_size, shuffle=shuffle,
                             nworkers=nworkers, imsize=imsize, randrot=randrot, scale=scale,
                             gksize=gksize, gdratio=gdratio, guse_pad=guse_pad,
                             gpad_factor=gpad_factor, aff_thresh=aff_thresh)


def synthtext_validloader(path, batch_size=16, shuffle=True, nworkers=4,
                        imsize=(224, 224), randrot=15, scale=0.5,
                        gksize=(35, 35), gdratio=2.0, guse_pad=False,
                        gpad_factor=0.1, aff_thresh=0.1):

    return _synthtext_loader(path, mode='train', batch_size=batch_size, shuffle=shuffle,
                             nworkers=nworkers, imsize=imsize, randrot=randrot, scale=scale,
                             gksize=gksize, gdratio=gdratio, guse_pad=guse_pad,
                             gpad_factor=gpad_factor, aff_thresh=aff_thresh)


if __name__ == '__main__':
    pass