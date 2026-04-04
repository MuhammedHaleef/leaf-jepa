
import torchvision.transforms as T

NORM_MEAN = [0.466726, 0.488969, 0.41028]
NORM_STD  = [0.195034, 0.170282, 0.213409]

def get_pretrain_transform():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
    ])

def get_finetune_transform(low_label: bool = False):
    base = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    return base  # CutMix/Mixup applied at batch level in training loop for low_label=True

def get_eval_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
