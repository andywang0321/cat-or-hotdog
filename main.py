import math
import torch
import os

from torchvision.utils import np

from src.model import ResNet
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

label_map: dict[int, str] = {0: "neither", 1: "cat", 2: "hot dog"}


def preprocess(path: str) -> torch.Tensor:
    image = Image.open(path).resize((64, 64))
    T = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return T(image)


def plot_results(images: list[torch.Tensor], labels: list[str]) -> None:
    num_imgs: int = len(images)
    ncols: int = math.ceil(math.sqrt(num_imgs))
    nrows: int = math.ceil(num_imgs / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx, (img_tensor, label) in enumerate(zip(images, labels)):
        ax = axes[idx]

        img = img_tensor.detach().cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(label, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # hide any extra subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def main():
    model = ResNet.to(device)
    model.load_state_dict(torch.load("ckpts/resnet.pt", map_location=device, weights_only=True))
    model.eval()
    print(f"Running {type(model).__name__} on {device}...")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    path: str = input("path/to/image or path/to/image_dir: ")
    assert os.path.exists(path), f"{path} does not exist!"

    images: list[torch.Tensor] = []
    image_paths: list[str] = (
        [path + "/" + img for img in os.listdir(path) if img[0] != "."] if os.path.isdir(path) else [path]
    )
    for image_path in image_paths:
        image: torch.Tensor = preprocess(image_path)
        images.append(image)
    img_batch: torch.Tensor = torch.stack(images).to(device)

    logits: torch.Tensor = model(img_batch)
    preds: list[int] = logits.argmax(dim=1).cpu().tolist()

    labels: list[str] = list(map(lambda p: label_map[p], preds))

    plot_results(images, labels)


if __name__ == "__main__":
    main()
