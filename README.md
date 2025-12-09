## 모델 개요 (Model Overview)

제안하는 모델은 'Enhanced-CNN(Convolutional Neural Network)' 구조로, 
TacTip 센서 내부의 핀 변형 이미지를 입력받아 접촉 위치(dX, dY, dZ)와 접촉 힘(F_x, F_y, F_z)의 6자유도 물리량을 동시에 추정하는 
다중 출력 회귀(Multi-output Regression) 모델임.

실시간 로봇 제어를 목표로 하므로, 
불필요하게 깊은 연산(ResNet 등)을 배제하고, 
특징 추출(Feature Extraction)과 회귀 예측(Regression)에 최적화된 경량화된 구조를 설계함.

수집된 데이터(https://github.com/Rottoda/data_collection)에서 확인할 수 있으며,
위 저장소로 수집된 데이터는 학습, 검증, 테스트 셋을 각각 [8:1:1] 비율로 분할하여 사용함. 

그 결과, 아래 표와 같이 위치뿐만 아니라 힘 예측에서도 R2가 0.98 이상의 높은 결정 계수를 달성함. 

| 구분 | 위치 X | 위치 Y | 위치 Z | 힘 X | 힘 Y | 힘 Z |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R2** | 0.9990 | 0.9982 | 0.9819 | 0.9901 | 0.9823 | 0.9872 |
| **MAE** | 0.0681 | 0.0652 | 0.0649 | 0.0105 | 0.0123 | 0.0196 |
| **MSE** | 0.0072 | 0.0107 | 0.0061 | 0.0002 | 0.0003 | 0.0007 |


---
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2023-00242528, RS-2024-00436182) and by the IITP (Institute for Information & Communications Technology Planning & Evaluation)-ITRC (Information Technology Research Center) grant funded by the Korea government (No. IITP-2025-RS-2024-00437756) and the other IITP grant funded by the Korea government (MSIT) (No. RS-2025-02263277).
