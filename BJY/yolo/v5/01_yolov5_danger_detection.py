import sys, time
import numpy as np
import torch
import cv2
# from ultralytics import YOLO
from collections import deque   # 고정길이 큐


def euclidean_dist(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

# 😎
def get_first_last_values(forklift_deque):
    """
    deque에서 None 값을 제외한 값들 중에서 가장 첫 값과 마지막 값을 구하는 함수
    - forklift_deque : 지게차 바운딩 박스의 좌표값이 저장된 deque 객체
    """
    # deque 내의 값이 충분하지 않을 경우 종료
    deque_len = len(forklift_deque)
    if deque_len <= 1:
        return
    
    front, back = 0, -1
    while True:
        value1 = forklift_deque[front]
        if value1 != None:
            break
        else:
            front += 1
    
    while True:
        value2 = forklift_deque[back]
        if value2 != None:
            break
        else:
            back -= 1
    
    # value1, value2 가 동일할 경우(deque 내에서 None이 아닌 값이 단 한 개) None 반환
    if front == back + deque_len:
        return
    
    return value1, value2


def extend_line(height, width, x1, y1, x2, y2):
    """
    forklift 의 최근 n개 프레임 정보를 사용해서 진행 방향을 구하고, 사진 상에서의 양 끝점을 구하는 함수
    n개 프레임 정보가 저장된 deque 내의 가장 첫 값과 끝 값을 사용해서 두 점을 잇는 직선의 양 끝 점을 구하여 반환한다.
    - height, width : 대상 이미지의 높이, 너비
    - x1, y1, x2, y2 : 직선을 그릴 때 사용할 두 점의 x, y 값
    """
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0: # 세로선
        return (int(x1), 0), (int(x1), height)
    elif dy == 0:   # 가로선
        return (0, int(y1)), (width, int(y1))
    else:
        grad = dy / dx
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
            return points[0], points[1]

# 😎
def calculate_route_coefs(x1, y1, x2, y2):
    """
    forklift 의 최근 n개 프레임 정보를 사용해서 진행 방향의 음함수 계수를 구하는 함수
    (x1, y1), (x2, y2) 값을 받아서 두 점을 잇는 직선을 구한다. (음함수 식 ax+by+c=0)
    """
    dx = x2 - x1
    dy = y2 - y1
    grad = dy / dx
    
    # 음함수 식 ax+by+c=0
    if dx == 0:
        a, b, c = 1, 0, -x1
    elif dy == 0:
        a, b, c = 0, 1, -y1
    else:
        a = grad
        b, c = -1, y1 - (a * x1)
    
    return a, b, c

# 😎
def detect_danger_between_forklift_and_person(forklift_deque, person_bbox):
    """ [여러 사람을 대상으로 작동할 수 있도록 수정 필요]
    forklift의 예상 진행 경로를 계산하고, 어떤 한 사람이 그 경로로부터 충분히 떨어져 있는지 판단하는 함수
    - 지게차의 종류 : (V), (D), (H)
    - forklift_deque : forklift의 바운딩 박스 좌표 여러 개를 저장하는 deque 객체
    - person_bbox : person의 바운딩 박스 좌표를 저장하는 리스트 객체
    """
    
    # deque에서 두 프레임의 좌표값 추출
    values = get_first_last_values(forklift_deque)
    if not values:
        return False
    else:
        value1, value2 = values
    x1, y1, w1, h1, _, cls1 = value1  # (xywh, conf, cls)
    x2, y2, w2, h2, _, cls2 = value2
    
    # x1, y1, _, _ = forklift_deque[0]
    # x2, y2, _, _ = forklift_deque[-1]
    
    coefs = calculate_route_coefs(x1, y1, x2, y2)
    a, b, c = coefs
    
    p_x1, p_y1, p_w1, p_h1 = person_bbox
    dist = abs(a * p_x1 + b * p_y1 + c) / (a**2 + b**2)**0.5

    # forklift_len = (w2**2 + h2**2)**0.5
    # person_len = (w1**2 + h2**2)**0.5
    
    tan_value = abs(a)  # 지게차 진행방향과 x축이 이루는 예각삼각형의 tangent 값
    cos_value = 1 / (1 + tan_value**2)**0.5
    sin_value = tan_value / (1 + tan_value**2)**0.5
    
    forklift_len = w2 * sin_value + h2 * cos_value      # 지게차의 진행방향과 수직인 직선에 지게차의 바운딩 박스를 정사영한 길이
    person_len = p_w1 * sin_value + p_h1 * cos_value    # 지게차의 진행방향과 수직인 직선에 사람의 바운딩 박스를 정사영한 길이
    
    ### 지게차의 방향에 따라 forklift_len의 길이에 가중치 적용 (일반적으로 H일 때 보다 V, D일 때 더 크게 잡히기 때문)
    weight_type = {1: 0.85, 3: 0.85, 4: 1.0}  # (V), (H) => 0.8 ~ 0.9
    weight = weight_type[int(cls2)]
    forklift_len = forklift_len * weight
    
    danger_cond1 = True if (forklift_len + person_len) * 0.5 >= dist else False
    
    ### 사람으로부터 가까워지는지 체크하는 코드
    dist1 = euclidean_dist(x1, y1, p_x1, p_y1)
    dist2 = euclidean_dist(x2, y2, p_x1, p_y1)
    danger_cond2 = True if (dist1 > dist2) else False
    
    # danger_flag
    return danger_cond1 & danger_cond2

## --------------------------------------------------------------------------------------
## 메인 코드
## --------------------------------------------------------------------------------------

# 커스텀 모델 불러오기
model = torch.hub.load('./yolov5', 'custom', path='./model/mnv_Model.pt', source='local') # ⭐

# 비디오 파일 로드
video_file = "./safe2.mp4"
cap = cv2.VideoCapture(video_file)

# 비디오 객체가 열렸는지 확인
if not cap.isOpened():
    print("Video open failed!")
    sys.exit()

# 최근 n개 프레임을 저장할 데크(deque) 객체 생성
DEQUE_MAXLEN = 5
forklift_frames = deque(maxlen=DEQUE_MAXLEN)
person_frames = deque(maxlen=DEQUE_MAXLEN)

# 프레임 간격 설정 (가변적)
frame_interval = 4

# forklift, person valid flag
forklift_valid, forklift_moves, person_valid = False, False, False # 초기값 : False

# 지게차 움직임의 기준치
MOVE_OR_NOT = 7

# 지게차 w(너비), h(높이) 최대값
# forklift_maxlen = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # (V), (D), (H) 😎 테스트용

# 1프레임씩 읽으며 위험상황 처리
while True:
    # time.sleep(0.2) # 😎 테스트용
    
    # 마지막에 적용할 cv2 사항들
    cv2_list = []
    
    # 비디오에서 현재 프레임의 위치
    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # 카메라의 ret, frame 값 가져오기
    # - ret : boolean (success or not)
    # - frame : image array vector
    ret, frame = cap.read()
    
    if not ret: 
        break
    
    if current_frame_pos % frame_interval == 0:
        # [1] YOLOv8 모델 적용
        # results : models.common.Detections
        print('[1] 실행')
        torch_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ⭐ (<= YOLOv5 핵심 코드들)
        results = model(torch_frame)  # ⭐
        
        # Visualize the results on the frame
        annotated_frames = results.render() # ⭐
        annotated_frame = cv2.cvtColor(annotated_frames[0], cv2.COLOR_BGR2RGB)  # ⭐
        
        detected_labels = results.pred[0][:, -1].int().tolist() # ⭐
        
        # [2] 지게차가 있는지 확인 (Trolly:2는 제외)
        if set(detected_labels) & set([1, 3, 4]):
            print('[2-1] 실행')
            indices = [i for i, x in enumerate(detected_labels) if x in [1, 3, 4]]  # 😎
            forklift_frame = results.xywh[0][indices[0]].clone()  # 😎 ⭐
            x_, y_, w_, h_, _, cls_ = forklift_frame  # 😎
            
            # 각 지게차 타입별 최대 대각선 길이 갱신
            # forklift_type = {1: 0, 3: 1, 4: 2}
            # idx = forklift_type[int(cls_)]
            # dist_ = euclidean_dist(0, 0, w_, h_)
            # if dist_ > forklift_maxlen[idx][2]:
            #     forklift_maxlen[idx][0] = w_
            #     forklift_maxlen[idx][1] = h_
            #     forklift_maxlen[idx][2] = dist_
            
            ### 중점좌표 보정 코드 (화면 양 끝에 있을 때)
            
            forklift_frames.append(forklift_frame)  # 😎 ⭐
            
            # 지게차가 움직이는 거리 계산, 움직이는지 여부 파악
            print(f'forklift_valid 값 : {forklift_valid}')
            print(f'forklift_frames 길이 : {len(forklift_frames)}')
            if (len(forklift_frames)==DEQUE_MAXLEN) and (forklift_frames.count(None) / len(forklift_frames) <= 0.5): # 😎😎
                print('[2-2] 실행')
                forklift_valid = True
                
                first_value, last_value = get_first_last_values(forklift_frames) # ⏳
                x1, y1, _, _, _, _ = first_value
                x2, y2, _, _, _, _ = last_value
                dist = euclidean_dist(x1, y1, x2, y2)   # 변위 계산
                forklift_moves = True if dist > MOVE_OR_NOT else False  # 변위가 기준치보다 크면 움직인다고 판단
        else:
            forklift_frames.append(None)
            if (len(forklift_frames)==DEQUE_MAXLEN) and (forklift_frames.count(None) / len(forklift_frames) > 0.5): # 😎😎
                # 트래킹 중인 지게차가 없어졌다면 데크 초기화
                if forklift_valid:
                    forklift_frames.clear()
                    forklift_valid = False
        
        # [3] 사람이 있는지 확인 (지게차가 있고 움직일 때)
        if forklift_valid and forklift_moves and (0 in detected_labels):
            print('[3-1] 실행')
            indices = [i for i, x in enumerate(detected_labels) if x == 0] # 😎
            person_frame = results.xywh[0][indices[0]][:-2].clone() # 😎 ⭐
            person_frame[1] += (person_frame[3] / 4)    # 사람 바운딩 박스 범위 조정(발 부분으로 한정)
            person_frame[3] = (person_frame[3] / 2)
            person_frames.append(person_frame)
            none_count = person_frames.count(None)
            if len(person_frames)==DEQUE_MAXLEN and none_count / len(person_frames) < 0.5:
                print('[3-2] 실행')
                person_valid = True
            else:
                person_valid = False
        else:
            person_valid = False
        
        # [4] 지게차 예상 진행 루트와의 직선 거리를 계산해서 위험여부를 알려줌
        if person_valid and detect_danger_between_forklift_and_person(forklift_frames, person_frame):  # ⏳
            print('[4] 실행')
            cv2_list.append(('collision risk occurred :o', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)) # 😎
            cv2_list.append((f'{person_frame}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)) # 테스트용
            print('collision risk occurred :o') # 테스트용
            # [GUI] if not mute : self.danger() # [위험상황 발생 시각 저장 기능]
        
        # [5] 지게차 예상 진행 루트(직선) 표시
        if forklift_valid and forklift_moves:
            print('[5-1] 실행')
            first_value, last_value = get_first_last_values(forklift_frames) # ⏳
            x1, y1, _, _, _, _ = first_value
            x2, y2, _, _, _, _ = last_value
            
            # 대상 사진의 높이, 너비
            height, width, _ = results.ims[0].shape
            
            if forklift_moves:
                print('[5-2] 실행')
                point1, point2 = extend_line(height, width, x1, y1, x2, y2) # 😎
                cv2_list.append((point1, point2, (0, 255, 0), 3)) # 😎
                dist = euclidean_dist(x1, y1, x2, y2)   # 빼도 되나?
                cv2_list.append((f'Dist : {dist:.3f}', (1030, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)) # 😎
        
        # cv2_list.append((f'forklift(V) => {forklift_maxlen[0]}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)) # 테스트용
        # cv2_list.append((f'forklift(D) => {forklift_maxlen[1]}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)) # 테스트용
        # cv2_list.append((f'forklift(H) => {forklift_maxlen[2]}', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)) # 테스트용
        
        # [6] cv2.putText, cv2.line 일괄 적용
        if len(cv2_list) > 0 :
                for k in range(0, len(cv2_list)):
                    if isinstance(cv2_list[k][0], str):
                        # 프레임, 텍스트 내용, 넣을 위치, 폰트 종류, 폰트 크기, 폰트 색, 폰트 굵기
                        cv2.putText(annotated_frame, cv2_list[k][0], cv2_list[k][1], cv2_list[k][2], cv2_list[k][3], cv2_list[k][4], cv2_list[k][5])
                    else:
                        # 프레임, 좌표1, 좌표2, 색상, 굵기
                        cv2.line(annotated_frame, cv2_list[k][0], cv2_list[k][1], cv2_list[k][2], cv2_list[k][3])
        
        # 이미지 크기 조정 (옵션)
        scale_factor = 0.8
        resized_frame = cv2.resize(annotated_frame, None, fx=scale_factor, fy=scale_factor)
        
        # [7] 어노테이션된 결과 프레임을 표시
        cv2.imshow('RESULT IMAGE', resized_frame)
        
        print(f'forklift_deque 값 : {forklift_frames}')
        print(f'forklift_valid 값 : {forklift_valid}')
        print('='*50)
    
    # 'q'가 눌리면 루프를 중단
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체를 해제하고 표시 창 닫기
cap.release()
cv2.destroyAllWindows()
