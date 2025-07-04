import cv2
import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from split_and_train_cnn import SimpleCNN  # 모델 구조 재사용

# 모델 경로
model_path = os.path.join(os.path.dirname(__file__), "cnn_model.pt")

# 모델 로딩
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 이미지 전처리 정의 (학습 시 사용한 것과 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 카메라 캡처
cap = cv2.VideoCapture(2)  # 2번 카메라

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 이진화 처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, 10
    )

    # PIL 변환 → 모델 예측
    pil_img = Image.fromarray(bin_img)
    input_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor).numpy()[0]
    dx, dy, dz = prediction

    # 결과 시각화
    show_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(show_img, f"dX: {dx:.3f}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 20, 147), 2)
    cv2.putText(show_img, f"dY: {dy:.3f}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 20, 147), 2)
    cv2.putText(show_img, f"dZ: {dz:.3f}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 20, 147), 2)

    # 이미지 보여주기
    cv2.imshow("Prediction", show_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
