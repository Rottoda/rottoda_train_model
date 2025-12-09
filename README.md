## 모델 개요 (Model Overview)

제안하는 모델은 'Enhanced-CNN(Convolutional Neural Network)' 구조로, 
TacTip 센서 내부의 핀 변형 이미지를 입력받아 접촉 위치(dX, dY, dZ)와 접촉 힘(F_x, F_y, F_z)의 6자유도 물리량을 동시에 추정하는 
다중 출력 회귀(Multi-output Regression) 모델임.

실시간 로봇 제어를 목표로 하므로, 
불필요하게 깊은 연산(ResNet 등)을 배제하고, 
특징 추출(Feature Extraction)과 회귀 예측(Regression)에 최적화된 경량화된 구조를 설계함.
