import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tabulate import tabulate

# === Model definition ===
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class FPFNPenalizedCELoss(nn.Module):
    def __init__(self, fn_weight=3.0, fp_weight=3.0):
        super().__init__()
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.ce = nn.CrossEntropyLoss(reduction='none')  # per-sample loss

    def forward(self, outputs, targets):
        # outputs: [batch_size, num_classes]
        # targets: [batch_size]
        loss = self.ce(outputs, targets)  # [batch_size]
        preds = torch.argmax(outputs, dim=1)

        # False Negatives: predicted 0, actual 1
        fn_mask = (preds == 0) & (targets == 1)
        loss[fn_mask] *= self.fn_weight

        # False Positives: predicted 1, actual 0
        fp_mask = (preds == 1) & (targets == 0)
        loss[fp_mask] *= self.fp_weight

        return loss.mean()

class UNetClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, features=[32, 64, 128, 256, 512]):
        super(UNetClassifier, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(features[-1]*2, num_classes)

    def forward(self, x):
        for down in self.downs:
            x = down(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# === Device config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transform and Data Loaders ===
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

train_dataset = datasets.ImageFolder(root="/home/bay/codes/unet/xray/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root="/home/bay/codes/unet/xray/val", transform=val_test_transform)
test_dataset = datasets.ImageFolder(root="/home/bay/codes/unet/xray/test", transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# === Dataset Info ===
print("Classes:", train_dataset.classes)
print("Class-to-idx:", train_dataset.class_to_idx)
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images: {len(test_dataset)}")

# === Model, Loss, Optimizer ===
model = UNetClassifier(in_channels=3, num_classes=2).to(device)
class_weights = torch.tensor([1.0, 5.0], dtype=torch.float).to(device)
criterion = FPFNPenalizedCELoss(fn_weight=3.0, fp_weight=3.0)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
print("Using device:", device)
print("Class weights:", class_weights)

# === Training + Validation ===
def train(num_epochs):
    print("\n[Starting Training]")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_running_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {acc:.4f}")
        if (epoch + 1) % 10 == 0:
            cm = confusion_matrix(val_labels, val_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)

            fig, ax = plt.subplots(figsize=(5, 5))
            disp.plot(ax=ax, cmap='Blues', colorbar=True)
            plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
            plt.show()

        # Free up memory
        torch.cuda.empty_cache()

    # Plot training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='x')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save model
    torch.save(model.state_dict(), "unet_classifier.pth")
    print("Model saved as 'unet_classifier.pth'")

# === Testing ===
def test():
    print("\n[Before Testing]")
    print(f"Test dataset contains {len(test_dataset)} images.")

    model.eval()
    all_preds, all_labels = [], []
    mislabelled_images = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Find misclassified images in the current batch
            for i in range(images.size(0)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                if true_label != pred_label:
                    # Compute the index of the image in the dataset
                    dataset_index = idx * test_loader.batch_size + i
                    image_path, _ = test_dataset.samples[dataset_index]
                    mislabelled_images.append((image_path, true_label, pred_label))

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")

    # === Confusion Matrix ===
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title("Confusion Matrix - Test Set")
    plt.show()

    # === Print mislabelled filenames ===
    print(f"\nTotal mislabelled images: {len(mislabelled_images)}")
    for path, true_label, pred_label in mislabelled_images:
        print(f"Mislabelled: {path} | True: {train_dataset.classes[true_label]} | Pred: {train_dataset.classes[pred_label]}")

# === Run Training + Save ===
print(f"Training with FN weight = {criterion.fn_weight}, FP weight = {criterion.fp_weight}")
train(num_epochs=100)

# === Show parameter ===
def print_training_config():
    model_name = model.__class__.__name__
    features = model.bottleneck.conv[0].in_channels // 2  # estimate base feature
    feature_list = [features * (2**i) for i in range(5)]  # approx. reconstruction

    config = [
        ["Model Architecture", model_name],
        ["Input Channels", model.downs[0].conv[0].in_channels],
        ["Output Classes", model.classifier.out_features],
        ["Feature Sizes", str(feature_list)],
        ["Loss Function", criterion.__class__.__name__],
        ["False Negative Weight", criterion.fn_weight],
        ["False Positive Weight", criterion.fp_weight],
        ["Optimizer", optimizer.__class__.__name__],
        ["Learning Rate", optimizer.param_groups[0]['lr']],
        ["Weight Decay", optimizer.param_groups[0]['weight_decay']],
        ["Batch Size", train_loader.batch_size],
        ["Epochs", num_epochs if 'num_epochs' in globals() else "Defined later"],
        ["Train Augmentations", ', '.join([t.__class__.__name__ for t in train_transform.transforms])],
        ["Val/Test Transforms", ', '.join([t.__class__.__name__ for t in val_test_transform.transforms])],
        ["Normalization Mean", str(train_transform.transforms[-1].mean)],
        ["Normalization Std", str(train_transform.transforms[-1].std)],
        ["Train Dataset Path", train_dataset.root],
        ["Val Dataset Path", val_dataset.root],
        ["Test Dataset Path", test_dataset.root],
        ["Device", str(device)],
        ["Model Save Path", "unet_classifier.pth"]
    ]

    print("\n[ Training Configuration Summary ]")
    print(tabulate(config, headers=["Parameter", "Value"], tablefmt="fancy_grid"))

print_training_config()

# === Load and Test ===
model.load_state_dict(torch.load("unet_classifier.pth"))
model.to(device)
print("Model loaded from 'unet_classifier.pth'")
test()