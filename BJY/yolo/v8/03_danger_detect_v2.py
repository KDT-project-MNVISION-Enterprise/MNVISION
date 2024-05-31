import sys, time
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from collections import deque

def extend_line(img, forklift_deque, color, thickness):
    """
    forklift 의 최근 n개 프레임 정보를 사용해서 진행 방향을 구하고, 사진 상에서의 양 끝점을 구하는 함수
    n개 프레임 정보가 저장된 deque 내의 가장 첫 값과 끝 값을 사용해서 두 점을 잇는 직선을 구한다.
    - forklift_deque : forklift의 바운딩 박스 좌표를 저장하는 deque 객체 
    """
    
    # deque 내의 값이 충분하지 않을 경우 종료
    deque_len = len(forklift_deque)
    if deque_len <= 1:
        return
    
    # 대상 사진의 높이, 너비
    height, width, _ = img.shape
    
    x1, y1, _, _ = forklift_deque[0]
    x2, y2, _, _ = forklift_deque[-1]
    
    dx = x2 - x1
    dy = y2 - y1
    grad = dy / dx
    
    if dx == 0: # 세로선
        cv2.line(img, (x1, 0), (x1, height), color, thickness, cv2.LINE_AA)
    elif dy == 0:   # 가로선
        cv2.line(img, (0, y1), (width, y1), color, thickness, cv2.LINE_AA)
    else:
        points = []
        
        # left border (x=0)
        y = y1 - x1 * grad
        if 0 <= y <= height:
            points.append((0, int(y)))
        
        # Right border (x=width)
        y = y1 + (width - x1) * grad
        if 0 <= y <= height:
            points.append((width, int(y)))
        
        # Top border (y=0)
        x = x1 - y1 / grad
        if 0 <= x <= width:
            points.append((int(x), 0))
        
        # Bottom border (y=height)
        x = x1 + (height - y1) / grad
        if 0 <= x <= width:
            points.append((int(x), height))
        
        if len(points) == 2:
            cv2.line(img, points[0], points[1], color, thickness, cv2.LINE_AA)


def calculate_route_coefs(forklift_deque):
    """
    forklift 의 최근 n개 프레임 정보를 사용해서 진행 방향의 음함수 계수를 구하는 함수
    deque 내의 가장 첫 값과 끝 값을 사용해서 두 점을 잇는 직선을 구한다. (음함수 식 ax+by+c=0)
    - forklift_deque : forklift의 바운딩 박스 좌표를 저장하는 deque 객체
    """
    
    # deque 내의 값이 충분하지 않을 경우 종료
    deque_len = len(forklift_deque)
    if deque_len <= 1:
        return
    
    x1, y1, _, _ = forklift_deque[0]
    x2, y2, _, _ = forklift_deque[-1]
    
    dx = x2 - x1
    dy = y2 - y1
    grad = dy / dx
    
    # 음함수 식 ax+by+c=0
    if dx == 0:
        a, b, c = 1, 0, -x1
    elif dy == 0:
        a, b, c = 0, 1, -y1
    else:
        a, b, c = grad, -1, y1 - (a * x1)
    
    return a, b, c


def detect_danger_between_forklift_and_person(forklift_deque, person_bbox):
    """ [여러 사람을 대상으로 작동할 수 있도록 수정 필요]
    forklift의 예상 진행 경로를 계산하고, 어떤 한 사람이 그 경로로부터 충분히 떨어져 있는지 판단하는 함수
    - forklift_deque : forklift의 바운딩 박스 좌표 여러 개를 저장하는 deque 객체
    - person_bbox : person의 바운딩 박스 좌표를 저장하는 리스트 객체
    """
    
    coefs = calculate_route_coefs(forklift_deque)
    if not coefs: 
        return
    
    a, b, c = coefs
    x1, y1, w1, h1 = person_bbox
    dist = abs(a * x1 + b * y1 + c) / (a**2 + b**2)**0.5
    
    _, _, w2, h2 = forklift_deque[-1]
    forklift_len = (w2**2 + h2**2)**0.5
    person_len = (w1**2 + h2**2)**0.5
    
    danger_flag = True if (forklift_len + person_len) * 0.5 >= dist else False
    return danger_flag


def detect_danger(cv2_texts, results, forklift_frames, forklift_valid):
    """
    사람-지게차 간 위험상황 감지 함수
    - cv2_texts : 위험상황 관련 cv2 표시할 텍스트 사항 모음 리스트
    - predict_frame : YOLOv8 모델을 적용하여 라벨링 된 프레임 (ndarray)
    - forklift_frames : forklift의 바운딩 박스 좌표를 저장하는 deque 객체 
    """
    
    # 지게차가 있는지 확인
    if 2 in results[0].boxes.cls:
        forklift_valid = True
        idx = results[0].boxes.cls.tolist().index(2)
        forklift_frames.append(results[0].boxes.xywh.tolist()[idx])
    else:
        forklift_valid = False
    
    # 사람이 있는지 확인
    if forklift_valid and (0 in results[0].boxes.cls):
        person_valid = True
        idx = results[0].boxes.cls.tolist().index(0)
        person_frame = results[0].boxes.xywh.tolist()[idx]
        # 지게차 예상 진행 루트와의 직선 거리를 계산해서 위험여부를 알려줌
        if detect_danger_between_forklift_and_person(forklift_frames, person_frame):
            # [위험상황 발생 시각 저장 기능] => 구현 예정
            cv2_texts.append(('collision risk detected', (10, 960), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255)))
            # cv2.putText(predict_frame, 'collision risk occurred', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            print('collision risk detected')
    else:
        person_valid = False
    
    # # 예상 진행 루트 표시 (직선)
    # if len(forklift_frames) >= 2:
    #     x1, y1, _, _ = forklift_frames[0]
    #     x2, y2, _, _ = forklift_frames[-1]
    #     extend_line(predict_frame, forklift_frames, (0, 255, 0), 3)
    #     # cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 10)
    #     dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
    #     cv2_texts.append((f'Dist : {dist:.3f}', (50, 990), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0)))
    #     # cv2.putText(predict_frame, f'Dist : {dist}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    
    return forklift_valid


### 메인 코드

# 커스텀 모델 불러오기
model = YOLO(r'./runs/detect/yolov8n_custom_1280x7204/weights/best.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPU 설정 (predict)
# model.to(DEVICE)

# 비디오 파일 로드
video_file = "./datasets/short.mp4"
cap = cv2.VideoCapture(video_file)

# 비디오 객체가 열렸는지 확인
if not cap.isOpened():
    print("Video open failed!")
    sys.exit()

# 최근 n개 프레임을 저장할 데크(deque) 객체 생성
forklift_frames = deque(maxlen=3)

# 프레임 간격 설정 (가변적)
frame_interval = 3

# forklift, person valid flag
forklift_valid, person_valid = False, False # 초기값 : False

# forklift, person count (일정 프레임 이상 존재할 경우 객체 탐지로 인정)
# forklift_cnt, person_cnt = 0, 0   # 아직 threshold 설정 안 함

# 1프레임씩 읽으며 위험상황 처리
while True:
    # 예측 프레임에 적용할 cv2 추가 사항 리스트
    cv2_labels = []
    
    # 트래킹 중인 지게차가 없다면 데크 초기화
    if not forklift_valid: #😏
        forklift_frames.clear() #😏
    
    # 비디오에서 현재 프레임의 위치
    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # 카메라의 ret, frame 값 가져오기
    # - ret : boolean (success or not)
    # - frame : image array vector
    ret, frame = cap.read()
    
    if not ret: 
        break
    
    ############################
    # yimsoyoung 호출 코드 위치 #
    ############################
    
    if current_frame_pos % frame_interval == 0: 
        # YOLOv8 모델 적용
        # results[0] : ultralytics.engine.results.Results
        results = model.predict(frame, conf=0.7)
        
        # Visualize the results on the frame
        predict_frame = results[0].plot()
        print(predict_frame)
        
        # 사람-지게차 간 위험감지 알고리즘 실행 
        forklift_valid = detect_danger(cv2_labels, results, forklift_frames, forklift_valid) #😏
    
    # 어노테이션 사항 적용 #😏
    for txt, loc, font_type, font_scale, color in cv2_labels:
        cv2.putText(predict_frame, txt, loc, font_type, font_scale, color)
    
    # 이미지 크기 조정
    scale_factor = 0.5
    resized_frame = cv2.resize(predict_frame, None, fx=scale_factor, fy=scale_factor)
    
    # 어노테이션된 프레임을 표시
    cv2.imshow('RESULT IMAGE', resized_frame)
    
    # 'q'가 눌리면 루프를 중단
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체를 해제하고 표시 창 닫기
cap.release()
cv2.destroyAllWindows()
