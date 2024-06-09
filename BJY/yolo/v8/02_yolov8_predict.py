# jupyer notebook 으로 동영상 예측을 실행하니 MemoryError가 발생해서 py 파일로 실행
# jupyter notebook에서 실행 결과(로그)가 너무 많이 표시될 때 문제가 발생하는 듯
# 파일 실행 위치 : yolo/v8/

import torch
import cv2
from ultralytics import YOLO

# 커스텀 모델 불러오기
model = YOLO(r'./runs/detect/yolov8n_custom_1280x7204/weights/best.pt')
# model = YOLO(r'./ysy_models/best.pt')

# GPU 설정 (predict)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 비디오 예측
video_file = "./datasets/Detect_test_Cam6.mp4"
results = model.predict(source= video_file, save=True, conf=0.6, device=0 if torch.cuda.is_available() else 'cpu')

