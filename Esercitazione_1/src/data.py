import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch

def get_data_transforms(config, is_training: bool = True):
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    augmentations = []
    if is_training and getattr(config, 'data_augmentation', None):
        aug_map = {
            "RandomRotation": transforms.RandomRotation(degrees=15),
            "RandomAffine": transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            "ColorJitter": transforms.ColorJitter(brightness=0.1, contrast=0.1)
        }
        for aug_name in config.data_augmentation:
            if aug_name in aug_map:
                augmentations.append(aug_map[aug_name])
            else:
                print(f"Warning: Unknown augmentation '{aug_name}', skipping.")
    return transforms.Compose(augmentations + base_transforms)

class TransformedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray, transform: transforms.Compose = None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    def __len__(self) -> int:
        return len(self.indices)
    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

def save_teacher_logits(model, dataloader, device, save_path):
    """Salva le predizioni (logits) del teacher su tutto il dataloader."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    torch.save(all_logits, save_path)
    return save_path

def load_teacher_logits(path):
    """Carica i logits salvati del teacher."""
    return torch.load(path)

class DistillationDataset(Dataset):
    """Dataset che restituisce (img, label, teacher_logits) per knowledge distillation."""
    def __init__(self, base_dataset, teacher_logits):
        self.base_dataset = base_dataset
        self.teacher_logits = teacher_logits
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        logits = self.teacher_logits[idx]
        return img, label, logits
