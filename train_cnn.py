import os
import shutil
import torch
import torch.nn as nn
import pandas as pd

from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import CenterCrop, Compose, RandomErasing, RandomResizedCrop, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import DogBreedTrainValDataset
from model import DogBreedResNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 50
    batch_size = 32
    start_epoch = 0
    best_acc = 0
    patience = 10

    labels_path = "labels.csv"
    train_dir = "train"

    train_transform = Compose([
    RandomResizedCrop(224, scale=(0.8, 1.0)),  # 👈 thay Resize
    RandomHorizontalFlip(),
    RandomRotation(15),
    ColorJitter(0.3, 0.3, 0.3),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225]),
    RandomErasing(p=0.25)
    ])

    val_transform = Compose([
    Resize(256),        # 👈 giữ tỉ lệ
    CenterCrop(224),    # 👈 crop giữa
    ToTensor(),
    Normalize([0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(labels_path)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["breed"]
    )

    train_dataset = DogBreedTrainValDataset(
        image_dir=train_dir,
        dataframe=train_df,
        transform=train_transform
    )

    val_dataset = DogBreedTrainValDataset(
        image_dir=train_dir,
        dataframe=val_df,
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    if os.path.isdir("tensorboard"):
        shutil.rmtree("tensorboard")

    if not os.path.isdir("training_models"):
        os.mkdir("training_models")

    writer = SummaryWriter("tensorboard")

    num_classes = len(train_dataset.class_to_idx)
    model = DogBreedResNet(num_classes=num_classes, pretrained=True).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.3)
    criterion = nn.CrossEntropyLoss()

    if os.path.isfile("training_models/last_resnet.pth"):
        ckpt = torch.load("training_models/last_resnet.pth", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_acc = ckpt.get("best_acc", 0)

    num_iters = len(train_dataloader)

    for epoch in range(start_epoch, num_epochs):
        model.train()

        progress_bar = tqdm(train_dataloader, colour="green")

        for iteration, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss_value = criterion(outputs, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_acc = (preds == labels).float().mean()

            progress_bar.set_description(
                f"Epoch {epoch+1}/{num_epochs}, Iteration {iteration+1}/{num_iters}, Loss {loss_value.item():.4f}, Acc {train_acc:.4f}"
            )

            writer.add_scalar("Train/Loss", loss_value.item(), epoch * num_iters + iteration)
            writer.add_scalar("Train/Accuracy", train_acc.item(), epoch * num_iters + iteration)    

        model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_predictions.extend(predictions.cpu().numpy().tolist())

        accuracy = accuracy_score(all_labels, all_predictions)
        writer.add_scalar("Validation/Accuracy", accuracy, epoch)

        scheduler.step(accuracy)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "class_to_idx": train_dataset.class_to_idx
        }

        torch.save(checkpoint, "training_models/last_resnet.pth")

        if accuracy > best_acc:
            best_acc = accuracy
            no_improve_epochs = 0
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, "training_models/best_resnet.pth")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy for {patience} consecutive epochs.")
            break    
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {accuracy:.4f}")
        
    writer.close()