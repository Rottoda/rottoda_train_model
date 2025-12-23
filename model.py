import torch
import torch.nn as nn
from torchvision import transforms

class KeypointUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(KeypointUNet, self).__init__()

         # 64x64 -> 32x32
        self.enc_conv1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 32x32 -> 16x16
        self.enc_conv2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 16x16 -> 8x8
        self.enc_conv3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck_conv = self.conv_block(256, 512)

        # 8x8 -> 16x16
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(512, 256) # 256(upconv) + 256(skip)

        # 16x16 -> 32x32
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(256, 128) # 128(upconv) + 128(skip)

        # 32x32 -> 64x64
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(128, 64) # 64(upconv) + 64(skip)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid())

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])])
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        e1 = self.enc_conv1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc_conv3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck_conv(p3)

        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1) 
        d3 = self.dec_conv3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec_conv2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec_conv1(d1)

        heatmap = self.heatmap_head(d1)

        return heatmap


class KeypointUNet256(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(KeypointUNet256, self).__init__()

        # --- Encoder (Contracting Path) ---
        # 256x256 -> 128x128
        self.enc_conv1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        # 128x128 -> 64x64
        self.enc_conv2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        # 64x64 -> 32x32
        self.enc_conv3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        # 32x32 -> 16x16 (추가된 레이어)
        self.enc_conv4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        # 16x16 -> 8x8 (추가된 레이어)
        self.enc_conv5 = self.conv_block(512, 1024)
        self.pool5 = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        # 8x8 크기에서 채널 두 배
        self.bottleneck_conv = self.conv_block(1024, 2048)

        # --- Decoder (Expansive Path) ---
        # 8x8 -> 16x16 (추가된 레이어)
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec_conv5 = self.conv_block(2048, 1024) # 1024(upconv) + 1024(skip e5)
        # 16x16 -> 32x32 (추가된 레이어)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv4 = self.conv_block(1024, 512) # 512(upconv) + 512(skip e4)
        # 32x32 -> 64x64
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(512, 256) # 256(upconv) + 256(skip e3)
        # 64x64 -> 128x128
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(256, 128) # 128(upconv) + 128(skip e2)
        # 128x128 -> 256x256
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(128, 64)   # 64(upconv) + 64(skip e1)

        # --- Output Head ---
        # dec_conv1의 출력 채널(64)을 받아 out_channels(1)로 변경
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid())
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])])
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        # Encoder
        e1 = self.enc_conv1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc_conv3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc_conv4(p3) # <--- 추가
        p4 = self.pool4(e4)     # <--- 추가
        
        e5 = self.enc_conv5(p4) # <--- 추가
        p5 = self.pool5(e5)     # <--- 추가

        # Bottleneck
        b = self.bottleneck_conv(p5) # p3가 아닌 p5를 입력

        # Decoder
        d5 = self.upconv5(b)            # <--- 추가
        d5 = torch.cat([d5, e5], dim=1) # <--- 추가
        d5 = self.dec_conv5(d5)         # <--- 추가
        
        d4 = self.upconv4(d5)           # <--- 추가 (d5 입력)
        d4 = torch.cat([d4, e4], dim=1) # <--- 추가
        d4 = self.dec_conv4(d4)         # <--- 추가

        d3 = self.upconv3(d4)           # <--- d4를 입력
        d3 = torch.cat([d3, e3], dim=1) 
        d3 = self.dec_conv3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec_conv2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec_conv1(d1)

        # Output
        heatmap = self.heatmap_head(d1)

        return heatmap