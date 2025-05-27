import time
from model import ResNet
from dataset import CatHotdogDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

batch_size = 64
learning_rate = 1e-4
num_epochs = 200

model_save_path = "ckpts/resnet.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(model, loader, loss_fn, optimizer):
    model.train()
    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits.squeeze(), labels.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        yield step, loss.item()


@torch.no_grad()
def eval(model, loader, loss_fn):
    model.eval()
    losses = []
    accuracies = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits.squeeze(), labels.squeeze())
        losses.append(loss.item())

        preds = logits.argmax(axis=1)
        correct = (preds == labels).sum().item()
        accuracy = correct / len(labels)
        accuracies.append(accuracy)
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def solver(model, train_loader, eval_loader, epochs):
    print(f"Training {type(model).__name__} on {device}...")

    model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=30)

    len_train_loader = len(train_loader)
    train_loss_history = []
    eval_loss_history = []

    for epoch in range(epochs):
        start = time.time()
        for step, train_loss in train(model, train_loader, loss_fn, optimizer):
            eval_loss, eval_acc = eval(model, eval_loader, loss_fn)
            scheduler.step(eval_loss)

            eval_loss_history.append(eval_loss)
            train_loss_history.append(train_loss)
        end = time.time()

        print(
            f"Epoch: {epoch + 1:2d}/{epochs},",
            f"Step: {step + 1:2d}/{len_train_loader},",
            f"Validation loss: {eval_loss:.2f},",
            f"Validation accuracy: {eval_acc:.2f}",
            f"LR: {1e4 * scheduler.get_last_lr()[0]:.2f}e-4",
            f"Time: {end - start:.1f}s",
        )
    return train_loss_history, eval_loss_history


def main():
    import plotext as plx

    loaders = {}

    for split in ("train", "eval", "test"):
        loaders[split] = DataLoader(
            CatHotdogDataset(f"data/{split}.zarr"),
            batch_size=batch_size,
            shuffle=True,
        )

    model = ResNet

    train_loss, eval_loss = solver(model, loaders["train"], loaders["eval"], num_epochs)

    plx.clf()
    plx.subplots(2, 1)
    plx.plot_size(80, 40)

    plx.subplot(1, 1)
    plx.plot(train_loss)
    plx.title("Train loss")

    plx.subplot(2, 1)
    plx.plot(eval_loss)
    plx.title("Eval loss")

    plx.show()

    test_loss, test_accuracy = eval(model, loaders["test"], nn.CrossEntropyLoss().to(device))
    print(f"Test loss: {test_loss:.3f},", f"Test accuracy: {test_accuracy:.2f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model checkpoint saved at {model_save_path}")


if __name__ == "__main__":
    main()
