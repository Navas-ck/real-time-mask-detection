import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # Dataset Paths
    # -----------------------------
    train_dir = r"C:\Users\nckmp\OneDrive\Desktop\projects\facemask detection\dataset\train"
    val_dir   = r"C:\Users\nckmp\OneDrive\Desktop\projects\facemask detection\dataset\val"
    test_dir  = r"C:\Users\nckmp\OneDrive\Desktop\projects\facemask detection\dataset\test"

    # -----------------------------
    # Strong Data Augmentation
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),

        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # -----------------------------
    # Load MobileNetV2
    # -----------------------------
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Modify final layer for 2 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    model = model.to(device)

    # -----------------------------
    # Loss & Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # -----------------------------
    # Training Loop (30 Epochs)
    # -----------------------------
    epochs = 20

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # -----------------------------
    # Test Accuracy
    # -----------------------------
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), "mask_detector.pth")
    print("🎉 Model saved as mask_detector.pth")