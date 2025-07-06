import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageToDisplacementDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.labels = pd.read_csv(csv_file)[["dX", "dY", "dZ"]].values
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [f"{i:03d}.png" for i in range(len(self.labels))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # 설정
    image_dir = "C:/Users/ASUS/Documents/Rottoda_TacTip/data_collection/images/session_20250705_002742/bin"
    csv_path = "C:/Users/ASUS/Documents/Rottoda_TacTip/data_collection/relative_random_points.csv"
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-3

    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # 데이터 준비
    dataset = ImageToDisplacementDataset(csv_path, image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델, 손실함수, 옵티마이저
    model = SimpleCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss / len(dataloader):.4f}")

    # 모델 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, "cnn_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ 모델 저장 완료: {model_save_path}")
