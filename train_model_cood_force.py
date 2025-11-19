import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.cm as cm

# 설정
image_dir = "C:/Users/ASUS/Documents/Rottoda_TacTip/data_collection/images/session_20250707_020701/bin"
csv_path = "C:/Users/ASUS/Documents/Rottoda_TacTip/data_collection/relative_random_points_v2.csv"
batch_size = 32
num_epochs = 50
learning_rate = 1e-3

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
      
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

dx_dy_dz_cols = ["dX", "dY", "dZ", "Fx", "Fy", "Fz"]
class ImageToDisplacementDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.labels = pd.read_csv(csv_file)[dx_dy_dz_cols].values
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [f"{i:05d}.png" for i in range(len(self.labels))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

full_dataset = ImageToDisplacementDataset(csv_file=csv_path, image_dir=image_dir, transform=transform)
all_indices = list(range(len(full_dataset)))
train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = SimpleCNN()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, "new_best_model.pt") 
    early_stopping = EarlyStopping(patience=100, verbose=True, path=model_save_path)


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] \tTrain Loss: {train_loss/len(train_loader):.4f} \tVal Loss: {test_loss/len(test_loader):.4f}")

        early_stopping(avg_test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f"Model saved to: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds.append(outputs.numpy())
            trues.append(labels.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)


    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds.append(outputs.numpy())
            trues.append(labels.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    for i, name in enumerate(["dX", "dY", "dZ", "Fx", "Fy", "Fz"]):
        r2 = r2_score(trues[:, i], preds[:, i])
        mae = mean_absolute_error(trues[:, i], preds[:, i])
        mse = mean_squared_error(trues[:, i], preds[:, i])
        print(f"{name} -> R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 5))
    axes = axes.flatten()
    titles = ['dX', 'dY', 'dZ', 'Fx', 'Fy', 'Fz']
    dX_values = trues[:, 0]
    colors = cm.Purples((dX_values - dX_values.min()) / (dX_values.max() - dX_values.min() + 1e-6))

    for i in range(6):
        axes[i].scatter(trues[:, i], preds[:, i], c=colors, s=30, edgecolor='k', alpha=0.8)
        axes[i].set_xlabel("Ground Truth")
        axes[i].set_ylabel("Prediction")
        axes[i].set_title(titles[i])
        axes[i].grid(True)

    plt.suptitle("Prediction vs Ground Truth by Sample (colored by dX intensity)")
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 5))
    axes2 = axes2.flatten()
    titles = ['dX', 'dY', 'dZ', 'Fx', 'Fy', 'Fz']
    for i in range(6):
        axes2[i].plot(trues[:, i], label='True', linewidth=2, linestyle='--')
        axes2[i].plot(preds[:, i], label='Predicted', linewidth=2)
        axes2[i].set_title(titles[i])
        axes2[i].set_xlabel("Sample Index")
        axes2[i].set_ylabel("Value")
        axes2[i].legend()
        axes2[i].grid(True)

    plt.suptitle("Validation Predictions vs Ground Truth (Line Plot)")
    plt.tight_layout()
    plt.show()