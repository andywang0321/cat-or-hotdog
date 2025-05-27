import zarr
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class CatHotdogDataset(Dataset):
    def __init__(self, path) -> None:
        root = zarr.open(path, mode="r")
        self.images = root["images"]
        self.labels = root["labels"]
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.preprocess(self.images[index])
        lab = torch.from_numpy(self.labels[index])
        return img, lab
