import torch
import torch.nn as nn
from datasets import ChordDataset
from networks import AlexNet, resnet18
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import plot_confusion_matrix, plot_logs, set_seed

## AlexNet Result
#               precision    recall  f1-score   support

#            C       1.00      0.94      0.97        18
#            D       0.92      0.92      0.92        13
#            G       0.95      1.00      0.98        20

#     accuracy                           0.96        51
#    macro avg       0.96      0.96      0.96        51
# weighted avg       0.96      0.96      0.96        51

## ResNet18 Result
#               precision    recall  f1-score   support

#            C       0.70      0.39      0.50        18
#            D       0.52      0.85      0.65        13
#            G       0.85      0.85      0.85        20

#     accuracy                           0.69        51
#    macro avg       0.69      0.70      0.67        51
# weighted avg       0.71      0.69      0.67        51

CONFIG = {
    "seed": 42,
    "data_root": "data/processed",
    "image_size": (224, 224),
    "batch_size": 32,
    "num_epochs": 25,
    "learning_rate": 1e-4,
    "num_workers": 4,
    "model_save_path": "training/best_model.pth",
}


def train_one_epoch(net, dataloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

        progress_bar.set_postfix(
            loss=running_loss / total_samples,
            acc=correct_predictions / total_samples,
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def evaluate(net, dataloader, criterion, device):
    net.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(
                loss=running_loss / total_samples,
                acc=correct_predictions / total_samples,
            )

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def main():
    set_seed(CONFIG["seed"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = ChordDataset(
        root=CONFIG["data_root"], image_size=CONFIG["image_size"]
    )
    train_size = int(0.8 * len(full_dataset))
    temp_size = len(full_dataset) - train_size
    test_size = val_size = temp_size // 2
    train_dataset, temp_dataset = random_split(full_dataset, [train_size, temp_size])
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )

    net = AlexNet(num_classes=3).to(device)
    # net = resnet18(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG["learning_rate"])

    best_val_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    print("Starting training...")
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")

        train_loss, train_acc = train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )
        print(
            f"Epoch {epoch+1} Training   -> Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
        )

        val_loss, val_acc = evaluate(net, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1} Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), CONFIG["model_save_path"])
            print(
                f"New best network saved to {CONFIG['model_save_path']} with validation accuracy: {best_val_acc:.4f}"
            )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    print("Evaluating on test set...")
    net.load_state_dict(torch.load(CONFIG["model_save_path"], weights_only=True))
    test_loss, test_acc = evaluate(net, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    print("Generating confusion matrix...")
    class_names = full_dataset.classes
    plot_confusion_matrix(net, test_loader, device, class_names)
    plot_logs(train_losses, train_accs, val_losses, val_accs)


if __name__ == "__main__":
    main()
