# jupyer notebook 으로 동영상 예측을 실행하니 MemoryError가 발생해서 py 파일로 실행
# jupyter notebook에서 실행 결과(로그)가 너무 많이 표시될 때 문제가 발생하는 듯

import torch
import cv2
from ultralytics import YOLO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 커스텀 모델 불러오기
model = YOLO(r'./runs/detect/yolov9c_custom_1280x7203/weights/best.pt')
# model = YOLO(r'./ysy_models/best.pt')

# GPU 설정 (predict)
model.to(DEVICE)

# 비디오 예측
video_file = "./datasets/Detect_test_Cam5.MP4_20240528_171759.mp4"
results = model.predict(source= video_file, save=True, conf=0.6, device=DEVICE)

