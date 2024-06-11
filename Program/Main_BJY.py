import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from ultralytics import YOLO
import numpy as np
import time
import os
import re
import pygame
import time
import threading
import csv
import datetime
import subprocess
from collections import deque
import torch
from qt_material import apply_stylesheet

### 본인의 작업환경에 맞게 파일경로 수정 필요 ###
# 임소영 ===========================================================
# test_filepath = r"test/video.mp4"
# mp3_file = r"MNVISION\Program\Audio\alarm_bell.mp3"
# form_class = uic.loadUiType(r"MNVISION/Program/UI/Video.ui")[0]
# test_filepath =r"C:\Users\mathn\Desktop\MNVISION\Program\Video\test2.mp4"
# mp3_file = "Program/Audio/alarm_bell.mp3"
# form_class = uic.loadUiType("C:\Users\mathn\Desktop\MNVISION\Program\UI\Video.ui")[0]
# ort_session = YOLO('Program/Model/best.onnx')
# ort_session2 = YOLO('Program/Model/best.onnx')
#==========================================================================


# 명노아=================================================================
test_filepath =r"C:\Users\mathn\Desktop\MNVISION\Program\Video\5번카메라_루프렉사이_진입.mp4"
mp3_file = "Program/Audio/alarm_bell.mp3"
form_class = uic.loadUiType("Program/UI/Video.ui")[0]
#ort_session = YOLO('Program/Model/best.onnx')
#ort_session2 = YOLO('Program/Model/best.onnx')
ort_session = torch.hub.load('Program/yolov5', 'custom', path='Program/Model/mnv_Model.pt', source='local')
ort_session2 = torch.hub.load('Program/yolov5', 'custom', path='Program/Model/mnv_Model.pt', source='local')
danger_detected = False
danger_delay = False
mute = False
#=========================================================================



class VideoProcessor:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.cap = None
        self.fps = None
        self.frame_count = None
        self.duration = None
        self.current_frame = 0
        self.is_playing = True

        if filepath:
            self.load_video(filepath)

    def load_video(self, filepath):
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
        return self.duration

    def get_frame(self):
        if self.cap is not None and self.is_playing:
            ret, frame = self.cap.read()
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 2
                return frame
            else:
                self.is_playing = False

    def set_frame_position(self, frame_number):
        self.current_frame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def release(self):
        if self.cap is not None:
            self.cap.release()

            
class ObjectDetection:
    def __init__(self, model):
        self.model = model
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.b_c = (0, 0, 255)      # blue
        self.g_c = (0, 255, 0)      # green
        self.y_c = (255, 255, 0)    # yellow
        self.w_c = (255, 255, 255)  # white
        self.thick = 2
        self.count = 1
        self.rack_count = 1
        self.result = False
        self.current_frame_pos = 0
        ##########################################
        ### 변주영 알고리즘
        self.DEQUE_MAXLEN = 3
        self.forklift_frames = deque(maxlen=self.DEQUE_MAXLEN) # 😎
        self.person_frames = deque(maxlen=self.DEQUE_MAXLEN) # 😎
        self.frame_interval = 3 # 프레임 간격 설정 (가변적)
        self.forklift_valid, self.forklift_moves, self.person_valid = False, False, False 
        self.MOVE_OR_NOT = 7 # 지게차 움직임의 기준치
        self.cv2_labels = [] # 마지막에 적용할 cv2 사항들
        ##########################################
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_file)
        global mute

    @staticmethod
    def play_music(file_path):
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
            
    def danger(self):
        self.result = True
        global danger_detected, danger_delay
        if not danger_detected and not danger_delay:
            danger_detected=True
            threading.Thread(target=self.play_music, args=(mp3_file,)).start()
            
    def transfer_two_points(self, data):
        X_values = data[:, 0]
        Y_values = data[:, 1]

        # 가장 작은 X와 가장 큰 X, 가장 작은 Y와 가장 큰 Y를 찾기
        min_X = np.min(X_values)
        max_X = np.max(X_values)
        min_Y = np.min(Y_values)
        max_Y = np.max(Y_values)

        return min_X, min_Y, max_X, max_Y
    

    ### 임소영 알고리즘 함수
    def yimsoyoung(self, list_ysy, class_ids, label, list_box, value, value2, cv2_list):
        upper_coordinates, lower_coordinates, u_x1, u_y1, u_x2, u_y2, l_x1, l_y1, l_x2, l_y2 = list_ysy
        x1, x2, y1, y2 = list_box
        f_x1, f_x2, f_y1, f_y2 = -1, -1, -1, -1
        
        # Forklift에 사람이 있는 경우 알림 표시
        if (2 in class_ids) or (3 in class_ids) or (4 in class_ids) :
            forklift_box = (0, 0, 0, 0)

            # Forklift가 감지될 때마다 박스를 기록
            if label in ['Forklift(H)', 'Forklift(V)', 'Forklift(D)']:
                forklift_box = (x1, x2, y1, y2) 

            if label == 'Person' : 
                X_MUL = 1.0
                Y_MUL = 1.0
                x2 = ((x1 + x2) / 2) * X_MUL
                y2 = ((y1 + y2) / 2) * Y_MUL
                x2 , y2 = int(x2), int(y2)
                
                if forklift_box : 
                    f_x1, f_x2, f_y1, f_y2 = forklift_box  # 수정된 부분
                    f_x1, f_x2 = f_x1 * X_MUL , f_x2 * X_MUL
                    f_y1, f_y2 = f_y1 * Y_MUL, f_y2 * Y_MUL
                    if (f_x1-50 <= x2 <= f_x2+50) and (f_y1-50 <= y2 <= f_y2+50):
                        cv2_list.append((0, 'Person on FORKLIFT', (600,350), self.font, 1, self.b_c, 3))
                        print('Person on FORKLIFT')
                        if not mute : self.danger()

                
        # Rack에 사람이 있는 경우 알림 표시
        if label == 'Person':
            X_MUL = 1.0 # 1.5
            Y_MUL = 1.0 # 0.5
            x2 = ((x1 + x2) / 2) * X_MUL
            y2 = ((y1 + y2) / 2) * Y_MUL
            x2 , y2 = int(x2), int(y2)
            txt_pt = (x1, y2 + 30)
            if upper_coordinates and (u_x1 <= x2 <= u_x2) and (u_y1 <= y2 <= u_y2):
                cv2_list.append((0, 'Person on UPPER RACK', txt_pt, self.font, 1, self.b_c, 3))
                print('Person on upper rack')
                if not mute : self.danger()
            if lower_coordinates and (l_x1 <= x2 <= l_x2) and (l_y1 <= y2 <= l_y2):
                cv2_list.append((0, 'Person on LOWER RACK', txt_pt, self.font, 1, self.b_c, 3))
                print('Person on lower rack')
                if not mute : self.danger()

        # Forklift가 Rack 공간에 작업 중인 경우 알림 표시 
        elif (label == 'Forklift(H)') or (label == 'Forklift(D)'):
            if upper_coordinates and ((x1 + x2) / 2 > (u_x1 + u_x2) / 2):
                # left
                d_1 = (u_x2 - x1) ** 2 + (u_y1 - y1) ** 2
                d_1 = np.sqrt(d_1)

                d_2 = (u_x2 - x2) ** 2 + (u_y2 - y2) ** 2
                d_2 = np.sqrt(d_2)

                value = (d_1 + d_2) / 2

            elif upper_coordinates and ((x1 + x2) / 2 < (u_x1 + u_x2) / 2):
                # right
                d_1 = (u_x1 - x2) ** 2 + (u_y1 - y1) ** 2
                d_1 = np.sqrt(d_1)

                d_2 = (u_x1 - x2) ** 2 + (u_y2 - y2) ** 2
                d_2 = np.sqrt(d_2)

                value = (d_1 + d_2) / 2

            if lower_coordinates and (x1 + x2) / 2 > (l_x1 + l_x2) / 2:
                # left
                d_3 = (l_x2 - x1) ** 2 + (l_y1 - y1) ** 2
                d_3 = np.sqrt(d_3)

                d_4 = (l_x2 - x2) ** 2 + (l_y2 - y2) ** 2
                d_4 = np.sqrt(d_4)

                value2 = (d_3 + d_4) / 2

            elif lower_coordinates and (x1 + x2) / 2 < (l_x1 + l_x2) / 2:
                # right
                d_3 = (l_x1 - x2) ** 2 + (l_y1 - y1) ** 2
                d_3 = np.sqrt(d_3)

                d_4 = (l_x1 - x2) ** 2 + (l_y2 - y2) ** 2
                d_4 = np.sqrt(d_4)

                value2 = (d_3 + d_4) / 2
            print(f"value : {value}   value2 : {value2}")

            if (value < 500) or (value2 < 500):
                if value < value2:
                    cv2_list.append((0, 'Folklift on UPPER RACK', (300,400), self.font, 1, self.y_c, 3))
                else:
                    cv2_list.append((0, 'Folklift on LOWER RACK', (300,400), self.font, 1, self.y_c, 3))
        else:
            self.count = 1

    def euclidean_dist(self, x1, y1, x2, y2):
        return ((x1-x2)**2 + (y1-y2)**2)**0.5
    
    def get_first_last_values(self, forklift_deque):
        """
        deque에서 None 값을 제외한 값들 중에서 가장 첫 값과 마지막 값을 구하는 함수
        - forklift_deque : 지게차 바운딩 박스의 좌표값이 저장된 deque 객체
        """
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
        
        return value1, value2
    
    
    def extend_line(self, height, width, x1, y1, x2, y2):
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


    def calculate_route_coefs(self, forklift_deque):
        """
        forklift 의 최근 n개 프레임 정보를 사용해서 진행 방향의 음함수 계수를 구하는 함수
        deque 내의 가장 첫 값과 끝 값을 사용해서 두 점을 잇는 직선을 구한다. (음함수 식 ax+by+c=0)
        - forklift_deque : forklift의 바운딩 박스 좌표를 저장하는 deque 객체
        """
        
        # deque 내의 값이 충분하지 않을 경우 종료
        deque_len = len(forklift_deque)
        if deque_len <= 1:
            return
        
        # deque에서 두 프레임의 좌표값 추출
        value1, value2 = self.get_first_last_values(forklift_deque)
        x1, y1, _, _ = value1
        x2, y2, _, _ = value2
        
        # x1, y1, _, _ = forklift_deque[0]
        # x2, y2, _, _ = forklift_deque[-1]
        
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


    def detect_danger_between_forklift_and_person(self, forklift_deque, person_bbox):
        """ [여러 사람을 대상으로 작동할 수 있도록 수정 필요]
        forklift의 예상 진행 경로를 계산하고, 어떤 한 사람이 그 경로로부터 충분히 떨어져 있는지 판단하는 함수
        - forklift_deque : forklift의 바운딩 박스 좌표 여러 개를 저장하는 deque 객체
        - person_bbox : person의 바운딩 박스 좌표를 저장하는 리스트 객체
        """
        
        coefs = self.calculate_route_coefs(forklift_deque)
        if not coefs: 
            return
        
        a, b, c = coefs
        p_x1, p_y1, p_w1, p_h1 = person_bbox
        dist = abs(a * p_x1 + b * p_y1 + c) / (a**2 + b**2)**0.5

        # _, _, w2, h2 = forklift_deque[-1]
        idx = -1
        while True:
            value = forklift_deque[idx]
            if value != None:
                _, _, w2, h2 = value
                break
            else:
                idx -= 1

        # forklift_len = (w2**2 + h2**2)**0.5
        # person_len = (w1**2 + h2**2)**0.5
        
        tan_value = abs(a)  # 지게차 진행방향과 x축이 이루는 예각삼각형의 tangent 값
        cos_value = 1 / (1 + tan_value**2)**0.5
        sin_value = tan_value / (1 + tan_value**2)**0.5
        
        forklift_len = w2 * sin_value + h2 * cos_value
        person_len = p_w1 * sin_value + p_h1 * cos_value
        
        danger_cond1 = True if (forklift_len + person_len) * 0.5 >= dist else False
        
        ### 사람으로부터 가까워지는지 체크하는 코드 (추가 예정)
        
        # danger_flag
        return danger_cond1


    def detect_danger(self, results, forklift_frames, forklift_valid, forklift_moves):
        """
        사람-지게차 간 위험상황 감지 함수
        - cv2_texts : 위험상황 관련 cv2 표시할 텍스트 사항 모음 리스트
        - predict_frame : YOLOv8 모델을 적용하여 라벨링 된 프레임 (ndarray)
        - forklift_frames : forklift의 바운딩 박스 좌표를 저장하는 deque 객체 
        """
        # print('detect_danger 실행됨')   # 😎
        
        detected_labels = results.pred[0][:, -1].int().tolist() # ⭐
        
        # [2] 지게차가 있는지 확인 (Trolly:2도 포함)
        if set(detected_labels) & set([1, 2, 3, 4]):
            # print('[2-1] 실행')
            indices = [i for i, x in enumerate(detected_labels) if x in [1, 2, 3, 4]]  # 😎
            forklift_frames.append(results.xywh[0][indices[0]][:-2].clone())  # 😎 ⭐
            none_count = forklift_frames.count(None)
            # print(f'forklift_valid 값 : {forklift_valid}')
            # print(f'forklift_frames 길이 : {len(forklift_frames)}')
            if (len(forklift_frames)==self.DEQUE_MAXLEN) and (none_count / len(forklift_frames) < 0.5):
                # print('[2-2] 실행')
                forklift_valid = True
                first_value, last_value = self.get_first_last_values(forklift_frames)
                x1, y1, _, _ = first_value
                x2, y2, _, _ = last_value
                dist = self.euclidean_dist(x1, y1, x2, y2)   # 변위 계산
                forklift_moves = True if dist > self.MOVE_OR_NOT else False  # 변위가 기준치보다 크면 움직인다고 판단
        else:
            forklift_frames.append(None)
            none_count = forklift_frames.count(None)
            if none_count / forklift_frames.count(None) >= 0.5:
                forklift_valid = False
        
        # [3] 사람이 있는지 확인 (지게차가 있고 움직일 때)
        if forklift_valid and forklift_moves and (0 in detected_labels):
            # print('[3-1] 실행')
            indices = [i for i, x in enumerate(detected_labels) if x == 0] # 😎
            person_frame = results.xywh[0][indices[0]][:-2].clone() # 😎 ⭐
            person_frame[1] += (person_frame[3] / 4)    # 사람 바운딩 박스 조정(발 부분으로 한정)
            person_frame[3] = (person_frame[3] / 2)
            self.person_frames.append(person_frame)
            none_count = self.person_frames.count(None)
            if len(self.person_frames)==self.DEQUE_MAXLEN and none_count / len(self.person_frames) < 0.5:
                # print('[3-2] 실행')
                person_valid = True # 😎
            else:
                person_valid = False # 😎
        else:
            person_valid = False
        
        # [4] 지게차 예상 진행 루트와의 직선 거리를 계산해서 위험여부를 알려줌
        if person_valid and self.detect_danger_between_forklift_and_person(forklift_frames, person_frame):
            # [위험상황 발생 시각 저장 기능] => GUI 팀 코드와 합치며 구현 예정
            # print('[4] 실행')
            self.cv2_labels.append((0, 'collision risk occurred :o', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)) # 😎
            self.cv2_labels.append((0, f'{person_frame}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)) # 테스트용
            # print('collision risk occurred :o') # 테스트용
            if not mute : self.danger() # 😎
        
        # [5] 지게차 예상 진행 루트(직선) 표시
        if forklift_valid and forklift_moves: # 😎
            # print('[5-1] 실행')
            first_value, last_value = self.get_first_last_values(forklift_frames) # 😎
            x1, y1, _, _ = first_value
            x2, y2, _, _ = last_value
            
            # 대상 사진의 높이, 너비
            height, width, _ = results.ims[0].shape # 😎
            
            if forklift_moves:
                # print('[5-2] 실행')
                point1, point2 = self.extend_line(height, width, x1, y1, x2, y2) # 😎
                self.cv2_labels.append((1, point1, point2, (0, 0, 255), 5)) # 😎
                dist = self.euclidean_dist(x1, y1, x2, y2)   # 빼도 되나?
                self.cv2_labels.append((0, f'Dist : {dist:.3f}', (1030, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)) # 😎
        
        return forklift_valid, forklift_moves
    
    def apply_model(self, frame, upper_coordinates=None, lower_coordinates=None):
        ### -------------------------------------------------------------------------
        ### 임소영 알고리즘 
        ### -------------------------------------------------------------------------
        self.cv2_labels = []    # cv2_labels 초기화 ⭐
        
        # self.current_frame_pos += 1     # 동영상 프레임 카운트
        # if self.current_frame_pos % self.frame_interval == 0 :
        #     return frame, self.result
        
        if upper_coordinates is None and lower_coordinates is None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ⭐
            results = self.model(frame) # ⭐
        else:
            if upper_coordinates:
                u_x1, u_y1, u_x2, u_y2 = self.transfer_two_points(np.array(upper_coordinates, dtype=np.int32))
                # frame = cv2.rectangle(frame, (u_x1, u_y1), (u_x2, u_y2), self.b_c, self.thick)
                frame = cv2.putText(frame, 'upper_rack', (u_x1, u_y1 - 10), self.font, 1, self.g_c, self.thick)
            else :
                u_x1, u_y1, u_x2, u_y2 = 0, 0, 0, 0

            if lower_coordinates:
                l_x1, l_y1, l_x2, l_y2 = self.transfer_two_points(np.array(lower_coordinates, dtype=np.int32))
                # frame = cv2.rectangle(frame, (l_x1, l_y1), (l_x2, l_y2), self.b_c, self.thick)
                frame = cv2.putText(frame, 'lower_rack', (l_x1, l_y1 - 10), self.font, 1, self.g_c, self.thick)
            else : 
                l_x1, l_y1, l_x2, l_y2 = 0, 0, 0, 0
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ⭐
            results = self.model(frame) # ⭐
            
            print('='*50)
            print(f"results \n{results}")

            # boxes = results.xyxy[0].cpu().numpy()   # 바운딩 박스 좌표 (x1, y1, x2, y2, confidence, class)
            # class_ids = results.xyxy[0][:, 5].cpu().numpy().astype(int)  # 클래스 아이디
            # print('='*50)
            # print(f"boxes {boxes}   class_ids {class_ids}")
            # print(f"boxes {type(boxes)}   class_ids {type(class_ids)}")
            # print(f"boxes {len(boxes)}   class_ids {len(class_ids)}")


            # 예측 결과에서 바운딩 박스와 클래스 아이디 추출
            results_data = results.xyxy[0]
            value = np.inf
            value2 = np.inf
            
            # 클래스 ID와 라벨을 리스트에 저장
            class_ids = []
            labels = []
            bboxes = []

            for *box, conf, cls_ in results_data:
                x1, y1, x2, y2 = map(int, box)  # 바운딩 박스 좌표
                class_id = int(cls_)             # 클래스 ID 
                classes = ['Person', 'Trolly', 'Forklift(H)', 'Forklift(D)', 'Forklift(V)']
                label = classes[class_id]

                class_ids.append(class_id)
                labels.append(label)
                bboxes.append([x1, y1, x2, y2])

                list_ysy = [upper_coordinates, lower_coordinates, u_x1, u_y1, u_x2, u_y2, l_x1, l_y1, l_x2, l_y2]
                list_box = [x1, y1, x2, y2]
                
                # 바운딩 박스를 그린다
                # self.cv2_labels.append((2, (x1, y1), (x2, y2), (255, 0, 0), 2))  # 😎
                # self.cv2_labels.append((0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2))   # 😎                
                
                ### 임소영
                self.yimsoyoung(list_ysy, class_ids, label, list_box, value, value2, self.cv2_labels)
                

        ### -------------------------------------------------------------------------
        ### 변주영 알고리즘 
        ### -------------------------------------------------------------------------
        # if (len(self.forklift_frames)==self.DEQUE_MAXLEN) and (not self.forklift_valid):
        #     self.forklift_frames.clear()    
        
        # self.current_frame_pos += 1     # 동영상 프레임 카운트
        # if self.current_frame_pos % self.frame_interval == 0 :
        #     self.forklift_valid, self.forklift_moves = self.detect_danger(results, self.forklift_frames, self.forklift_valid, self.forklift_moves)
        
        
        ### -------------------------------------------------------------------------
        ### 변주영 결과 + 임소영 결과 => frame 위에 opencv로 표시 
        ### -------------------------------------------------------------------------
        annotated_frames = results.render() # ⭐
        annotated_frame = cv2.cvtColor(annotated_frames[0], cv2.COLOR_BGR2RGB)  # ⭐
        
        if len(self.cv2_labels) > 0 :
            for k in range(0, len(self.cv2_labels)):
                cv2_type = self.cv2_labels[k][0]
                if cv2_type == 0:
                    # 프레임, 텍스트 내용, 넣을 위치, 폰트 종류, 폰트 크기, 폰트 색, 폰트 굵기
                    cv2.putText(annotated_frame, self.cv2_labels[k][1], self.cv2_labels[k][2], self.cv2_labels[k][3], self.cv2_labels[k][4], self.cv2_labels[k][5], self.cv2_labels[k][6])
                elif cv2_type == 1:
                    # 프레임, 좌표1, 좌표2, 색상, 굵기
                    cv2.line(annotated_frame, self.cv2_labels[k][1], self.cv2_labels[k][2], self.cv2_labels[k][3], self.cv2_labels[k][4])
                elif cv2_type == 2:
                    cv2.rectangle(annotated_frame, self.cv2_labels[k][1], self.cv2_labels[k][2], self.cv2_labels[k][3], self.cv2_labels[k][4])

        # cv2.putText(frame, 'Object Detection With YOLOv8', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return annotated_frame, self.result    # ⭐


class CameraProcessor:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            
            raise ValueError("Error: Could not open camera.")
        self.frame = None
        self.model_flag = False

    def apply_model(self, frame, model):
        results = model(frame)
        return results[0].plot()


    def get_frame(self):
        ret, self.frame = self.cap.read()
        return self.frame if ret else None

    def release(self):
        if self.cap is not None:
            self.cap.release()


class FrameSaver:
    def __init__(self):
        self.frames = []
        self.range_num = 3

    def save_frame(self, frame):
        self.frames.append(frame.copy())
        if len(self.frames) > self.range_num*30 :
            del self.frames[0]

    def save_to_video(self, output_path, fps=15):
        if len(self.frames) == 0:
            raise ValueError("Error: No frames to save.")
        frame_height, frame_width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        start_time = time.time()
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        for frame in self.frames:
            out.write(frame)
            if time.time() - start_time >= self.range_num*2 :
                    break
        out.release()

class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.current_frame = None
        self.setMouseTracking(True)
        self.drawing = False

    def setFrame(self, frame):
        self.current_frame = frame
        self.update()

    def getFrameSize(self):
        if self.current_frame is not None:
            return self.current_frame.shape[1], self.current_frame.shape[0]  # (width, height)
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and len(self.points) < 4:
            self.points.append(event.pos())
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_frame is not None:
            image = QImage(self.current_frame.data, self.current_frame.shape[1], self.current_frame.shape[0], self.current_frame.strides[0], QImage.Format_RGB888)
            painter = QPainter(self)
            painter.drawImage(0, 0, image)
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            for point in self.points:
                painter.drawEllipse(point, 5, 5)
            if len(self.points) == 4:
                sorted_points = self.sortPoints(self.points)
                pen.setColor(Qt.blue)
                painter.setPen(pen)
                painter.drawPolygon(*sorted_points)

    def sortPoints(self, points):
        points = sorted(points, key=lambda p: (p.x(), p.y()))
        top_points = sorted(points[:2], key=lambda p: p.y())
        bottom_points = sorted(points[2:], key=lambda p: p.y())
        return top_points[0], top_points[1], bottom_points[1], bottom_points[0]

class SelectAreaDialog(QDialog):
    def __init__(self, video_path=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Area')
        self.video_widget = VideoWidget()
        self.setWindowState(Qt.WindowMaximized)
        self.coordinates = None
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_widget)
        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(video_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(30)

    def next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_widget.setFrame(frame)
            if not self.isVisible():  
                frame_size = self.video_widget.getFrameSize()
                if frame_size:
                    self.resize(frame_size[0], frame_size[1])
                self.show()
        else:
            self.timer.stop()

    def mouseReleaseEvent(self, event):
        if len(self.video_widget.points) == 4:
            sorted_points = self.video_widget.sortPoints(self.video_widget.points)
            self.coordinates = [(p.x(), p.y()) for p in sorted_points]
            self.accept()  # Close the dialog and emit accepted signal
        else:
            super().mouseReleaseEvent(event)  # Call the base class implementation

    def closeEvent(self, event):
        self.reject()  # Emit the rejected signal when the dialog is closed

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)

        self.video_processor = VideoProcessor()
        self.camera_processor = CameraProcessor(camera_index = test_filepath) # on_dialog_finished 추가 작업 필요
        self.frame_saver = FrameSaver()
        self.video_widget = VideoWidget()
        self.model = ObjectDetection(ort_session)

        self.model_flag = False
        self.rectangle1_flag = False
        self.rectangle2_flag = False
        self.delay_term=False
        self.points1 = []
        self.points2 = []

        self.skip_num = 5
        self.btn_start_detection.clicked.connect(self.toggle_model)
        self.btn_pause.clicked.connect(self.start_video)
        self.btn_pause.hide()
        self.Current.hide()
        self.Total_length.hide()
        self.Video_bar.hide()
        self.btn_forward.hide()
        self.btn_prev.hide()
        self.btn_stop_start.hide()
        self.Video_upload.clicked.connect(self.load_video_file)
        self.btn_stop_start.clicked.connect(self.toggle_play_pause)
        self.btn_forward.clicked.connect(self.forward)
        self.btn_prev.clicked.connect(self.backward)
        self.Video_bar.valueChanged.connect(self.slider_moved)
        self.rack_btn_1.clicked.connect(lambda: self.open_select_area_dialog(1))
        self.rack_btn_2.clicked.connect(lambda: self.open_select_area_dialog(2))
        self.rack_btn_3.clicked.connect(lambda: self.draw_rectangle(1))
        self.rack_btn_4.clicked.connect(lambda: self.draw_rectangle(2))

        self.checkBox2.clicked.connect(self.muting)
        self.comboBox.currentIndexChanged.connect(self.set_save_frame_sec)
        self.comboBox_2.currentIndexChanged.connect(self.set_skip_sec)
        
        for i in range(1, 10):
            self.Log_text.addItem(f"00:0{i}:00")
        self.Log_text.itemClicked.connect(self.item_clicked)
        self.Log_text_2.itemClicked.connect(self.play_video)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.timer_video = QTimer(self)
        self.timer_video.timeout.connect(self.process_video)
        self.timer.start(15)
        
    def add_red_overlay(self):
        red_overlay = QGraphicsRectItem(0, 0, self.on_air_camera.width(), self.on_air_camera.height())
        red_overlay.setBrush(QBrush(QColor(255, 0, 0, 127)))  # Red color with 50% transparency
        self.scene2.addItem(red_overlay)

    def toggle_red_overlay(self):
        global danger_detected
        danger_detected=not danger_detected

    def danger_run(self):
        self.danger_timer = QTimer()
        self.danger_timer.timeout.connect(self.toggle_red_overlay)
        self.danger_timer.start(500)
        QTimer.singleShot(3100, self.stop_timer)

    def stop_timer(self):
        self.danger_timer.stop()
        global danger_detected
        danger_detected=False
        
    def play_video(self, item):
        video_filename = f"{item.text()}.mp4"
        
        # 현재 실행 위치의 경로를 가져옴
        current_directory = os.getcwd()
        
        # 비디오 파일의 전체 경로
        video_filepath = os.path.join(current_directory, video_filename)
        
        # VLC 미디어 플레이어로 비디오 파일 재생 (경로는 VLC가 설치된 위치에 따라 다를 수 있음)
        vlc_path = "C:\\Program Files\\VideoLAN\\VLC\\vlc.exe"  # VLC 실행 파일의 경로
        
        # 비디오 파일이 존재하는지 확인
        if os.path.isfile(video_filepath):
            # VLC로 비디오 파일 재생
            subprocess.run([vlc_path, video_filepath])
        else:
            print(f"비디오 파일 '{video_filepath}'을(를) 찾을 수 없습니다.")

    
    
    def draw_rectangle(self,num):
        if num==1:
            text = self.rack_text_1.toPlainText()
            print(text)
            pattern = r'\((\d+), (\d+)\)'
            matches = re.findall(pattern, text)
            self.points1 = [(int(x), int(y)) for x, y in matches]
            self.rectangle1_flag = not self.rectangle1_flag
        if num==2:
            text = self.rack_text_2.toPlainText()
            pattern = r'\((\d+), (\d+)\)'
            matches = re.findall(pattern, text)
            self.points2 = [(int(x), int(y)) for x, y in matches]
            self.rectangle2_flag = not self.rectangle2_flag
        
    def open_select_area_dialog(self, num):
        if num==1:
            self.rectangle1_flag = False
        if num==2:
            self.rectangle2_flag = False
        self.coordinate_box=num
        self.camera_processor.cap.release() # 메모리 해제 
        self.dialog = SelectAreaDialog(test_filepath, self)
        self.dialog.finished.connect(self.on_dialog_finished)
        self.dialog.show()
        

    def on_dialog_finished(self, result):
        if result == QDialog.Rejected:
            pass
        if self.coordinate_box == 1 and self.dialog.coordinates :
            self.rack_text_1.setText(str(self.dialog.coordinates))
            self.draw_rectangle(1)
        elif self.coordinate_box == 2 and self.dialog.coordinates:
            self.rack_text_2.setText(str(self.dialog.coordinates))
            self.draw_rectangle(2)
        else:
            print("Invalid coordinate box number") 
        self.dialog.deleteLater()  # 다이얼로그 객체 삭제
        self.camera_processor.cap = cv2.VideoCapture(test_filepath)
        
            
    def load_video_file(self):
        self.btn_pause.hide()
        self.Current.hide()
        self.Total_length.hide()
        self.Video_bar.hide()
        self.btn_forward.hide()
        self.btn_prev.hide()
        self.btn_stop_start.hide()
        fname = QFileDialog.getOpenFileName(self, 'Open file', './', 'Video files (*.mp4 *.avi)')
        if fname[0]:
            duration = self.video_processor.load_video(fname[0])
            self.Total_length.setText(self.transform_time(duration))  
            self.display_first_frame()
            self.display_video_info()
            self.btn_pause.show()

    def display_first_frame(self):
        frame = self.video_processor.get_frame()
        if frame is not None:
            self.show_img(self.graphicsView, self.scene, frame)

    def display_video_info(self):
        info_text = (f"파일명: {os.path.basename(self.video_processor.filepath)}\n"
                     f"최근 수정 날짜 : {datetime.datetime.fromtimestamp(os.path.getmtime(self.video_processor.filepath)).strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"프레임: {self.video_processor.frame_count}\n"
                     f"크기: {self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n"
                     f"영상 길이: {str(datetime.timedelta(seconds=int(self.video_processor.duration)))}\n"
                     f"fps: {self.video_processor.fps}")
        self.Video_info_text.setText(info_text)

    def transform_time(self, time_input):
        if isinstance(time_input, float):
            return f"{int(time_input // 3600):02}:{int((time_input % 3600) // 60):02}:{int(time_input % 60):02}"
        elif isinstance(time_input, str) and len(time_input.split(':')) == 3:
            hours, minutes, seconds = map(int, time_input.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        raise ValueError("Invalid input format. Provide an integer or a 'HH:MM:SS' string.")

    def start_video(self):
        self.timer_video.start(30)
        self.btn_pause.hide()
        self.Current.show()
        self.Total_length.show()
        self.Video_bar.show()
        self.btn_forward.show()
        self.btn_prev.show()
        self.btn_stop_start.show()

    def process_video(self):
        frame = self.video_processor.get_frame()
        if self.video_processor.is_playing: # 동영상 모델 적용 상시
            frame = self.camera_processor.apply_model(frame, ort_session2)
            self.show_img(self.graphicsView, self.scene, frame)
            self.update_current_time()

    def toggle_model(self):
        self.model_flag = not self.model_flag
        self.btn_start_detection.setText("모델 적용 중" if self.model_flag else "탐지 시작")

    def update_frame(self):
        frame = self.camera_processor.get_frame()
        
        if frame is not None:
            if self.model_flag :
                if self.points1 and self.points2:
                    frame, result = self.model.apply_model(frame, self.points1, self.points2)

                elif self.points1 :
                    frame, result = self.model.apply_model(frame, self.points1)
                elif self.points2 :
                    frame, result = self.model.apply_model(frame,lower_coordinates=self.points2)
                else :
                    frame, result = self.model.apply_model(frame)
                if result :
                    if not self.delay_term:
                        time = self.dialog_open()
                        self.Log_text_2.addItem(time)
                        self.delay_term = True
                        self.danger_run()
                        global danger_delay
                        danger_delay = True
                        threading.Timer(7, self.reset_delay_term).start()
                     
            if self.rectangle1_flag:
                points_int = np.array(self.points1, dtype=np.int32)
                cv2.polylines(frame, [points_int], True, (255, 0, 0), thickness=2)
            if self.rectangle2_flag:
                points_int = np.array(self.points2, dtype=np.int32)
                cv2.polylines(frame, [points_int], True, (255, 0, 0), thickness=2)

            self.show_img(self.on_air_camera, self.scene2, frame)
            self.frame_saver.save_frame(frame)
            self.video_widget.setFrame(frame)
            
    def reset_delay_term(self):
        self.delay_term = False
        global danger_delay
        danger_delay=False
        
        global danger_detected
        danger_detected = False
        print("10초가 지나서 delay_term이 False로 변경되었습니다.")

    def show_img(self, element, scene, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(element.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scene.clear()
        scene.addPixmap(scaled_pixmap)
        element.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        if danger_detected:
            self.add_red_overlay()

    def slider_moved(self, position):
        frame_position = int(position * self.video_processor.frame_count / 100)
        self.video_processor.set_frame_position(frame_position)
        self.update_current_time()

    def item_clicked(self, item):
        frame_position = self.transform_time(item.text()) * self.video_processor.fps
        self.video_processor.set_frame_position(frame_position)
        self.Video_bar.setValue(int(frame_position / self.video_processor.frame_count * 100))

    def toggle_play_pause(self):
        self.video_processor.is_playing = not self.video_processor.is_playing
        icon = QIcon('Program/video_icon/play.png' if not self.video_processor.is_playing else 'Program/video_icon/pause2.png') #
        self.btn_stop_start.setIcon(icon)

    def forward(self):
        self.video_processor.set_frame_position(self.video_processor.current_frame + int(self.skip_num * self.video_processor.fps))
        self.Video_bar.setValue(int(self.video_processor.current_frame / self.video_processor.frame_count * 100))

    def backward(self):
        self.video_processor.set_frame_position(self.video_processor.current_frame - int(self.skip_num * self.video_processor.fps))
        self.Video_bar.setValue(int(self.video_processor.current_frame / self.video_processor.frame_count * 100))

    def update_current_time(self):
        self.Current.setText(self.transform_time(self.video_processor.current_frame / self.video_processor.fps))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.timer.stop()
            self.timer_video.stop()
            self.video_processor.release()
            self.camera_processor.release()
            event.accept()
            sys.exit()
        else:
            event.ignore()

    def play_saved_frames(self):
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        output_path = f'{timestamp}.mp4'
        frame_delay = 1 / 30

        try:
            self.frame_saver.save_to_video(output_path)
        except ValueError as e:
            print(e)
            return
        # for frame in self.frame_saver.frames:
        #     #self.show_img(self.play_frame_view, self.scene3, frame)
        #     self.show_img(self.on_air_camera, self.scene2, frame)
        #     if danger_detected:
        #         self.add_red_overlay()
        #     QCoreApplication.processEvents()
        #     time.sleep(frame_delay)
        return timestamp

    def dialog_open(self):
        # self.dialog = QDialog()
        # self.dialog.setWindowTitle('영상 저장')
        # self.play_frame_view = QGraphicsView(self.dialog)
        # self.scene3 = QGraphicsScene()
        # self.play_frame_view.setScene(self.scene3)

        # dialog_layout = QVBoxLayout()
        # dialog_layout.addWidget(self.play_frame_view)
        # self.dialog.setLayout(dialog_layout)

        # self.message_label = QLabel("동영상 저장 중", self.dialog)
        # self.message_label.setAlignment(Qt.AlignCenter)
        # self.message_label.setStyleSheet("QLabel {font-size: 24px; font-weight: bold; }")
        # dialog_layout.addWidget(self.message_label)

        # screen = QDesktopWidget().screenGeometry()
        # width = screen.width() // 2
        # height = screen.height() // 2
        # self.dialog.resize(width, height)

        # self.dialog.show()
        time = self.play_saved_frames()
        #self.message_label.setText("동영상 저장 완료")
        return time

    def muting(self) :
        global mute
        if self.checkBox2.isChecked() :
            self.checkBox2.setText("On")
            mute=True
        else :
            self.checkBox2.setText("Off")
            mute=False

    def set_save_frame_sec(self) :
        select = self.comboBox.currentText()
        num = re.findall(r'-?\d+', select)
        self.frame_saver.range_num = int(num[0])

    def set_skip_sec(self) :
        select = self.comboBox_2.currentText()
        num = re.findall(r'-?\d+', select)
        self.skip_num = int(num[0])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    apply_stylesheet(app, theme="dark_teal.xml", css_file='Program/video_icon/custom.css')
    myWindow.show()
    sys.exit(app.exec_())