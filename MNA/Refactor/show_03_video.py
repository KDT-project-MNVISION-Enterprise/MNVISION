# ==================================================================
# 알림 신호를 위한 랙의 좌표값 설정하기
# - GUI에서 좌표값을 받아야함!
# ==================================================================
import cv2, time
import numpy as np
from ultralytics import YOLO

# 일단 고정되어있다고 가정하자
# u_x1, u_y1, u_x2, u_y2 = 137, 32, 437, 189     # cam 5
# d_x1, d_y1, d_x2, d_y2 = 110, 253, 363, 515
u_x1, u_y1, u_x2, u_y2 = 866, 33, 1111, 294      # cam 6
l_x1, l_y1, l_x2, l_y2 = 897, 300, 1232, 550


# 저장된 모델 불러오기
num = 7
model = YOLO("./runs/detect/train7/weights/best.pt")


# 비디오 파일 로드
file = "C:/Users/kdp/Desktop/test_video_01.mp4"
video = cv2.VideoCapture(file)


# 프레임 간격 설정 (예: 매 10번째 프레임 읽기)
frame_interval = 5

# opencv 관련 설정값
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
b_c = (0, 0, 255)      # blue
g_c = (0, 255, 0)      # green
y_c = (255, 255, 0)    # yellow
w_c = (255, 255, 255)  # white
thick = 2

# 비디오 프레임 처리
count = 1
sec = 3
rack_count = 1

# 새로운 비디오 파일 초기화 -> 저장용
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(video.get(cv2.CAP_PROP_FPS))
# out = cv2.VideoWriter(f'{file[:-4]}_output.avi', fourcc, fps, (width, height))



while True:
    current_frame_pos = int(video.get(cv2.CAP_PROP_POS_FRAMES))

    ret, frame = video.read()

    if not ret:
        time.sleep(sec)
        video.release()
        # out.release()
        cv2.destroyAllWindows()
        print('can NOT read video. It will be stopped.')

        break

    if current_frame_pos % frame_interval == 0 :

        # frame 위에 랙 구역 표시하기 

        frame = cv2.rectangle(frame, (u_x1, u_y1), (u_x2, u_y2), g_c, thick)
        frame = cv2.putText(frame, 'upper_rack', (u_x1, u_y1-10), font, 1, g_c, thick)
        frame = cv2.rectangle(frame, (l_x1, l_y1), (l_x2, l_y2), g_c, thick)
        frame = cv2.putText(frame, 'lower_rack', (l_x1, l_y1-10), font, 1, g_c, thick)
        
        # YOLOv8 모델 적용
        # results = model(frame)
        results = model.predict(frame, conf = 0.7)


        # Run batched inference on a list of images
        # results = model(["im1.jpg", "im2.jpg"])  # return a list of Results objects

        # Process results list
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     probs = result.probs  # Probs object for classification outputs
        #     obb = result.obb  # Oriented boxes object for OBB outputs
        #     result.show()  # display to screen
        #     result.save(filename="result.jpg")  # save to disk



        # print('================================ results ===================================')
        # print(len(results))
        # print('==================================== 0 ========================================')
        # print(results[0])
        # print('===============================================================================')

        # 잡히는 frame마다 결과 처리 
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            speeds = result.speed
            probs = result.probs
            obb = result.obb

            for box, class_id in zip(boxes, class_ids):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 좌표값을 정수로 변환

                label = model.names[class_id]
                print(f"좌표: ({x1}, {y1}) - ({x2}, {y2})  라벨: {label}")
                print(f"속도: {speeds}")
                print(f"확률: {probs}    obb : {obb}")

                # 탐지된 객체에 대해 바운딩 박스를 표시하기
                cv2.rectangle(frame, (x1, y1), (x2, y2), y_c, thick)
                cv2.putText(frame, label, (x1, y1-10), font, 1, y_c, thick)

                # 뮤팅 알림 표시 
                if label == 'Person' :
                    count += 1
                    print(f"count : {count}")

                    if count > 5 : 
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), g_c, 2)
                        # cv2.putText(frame, label, (x1, y1-10), font, 1, (36,255,12), 2)

                        # upper rack
                        if (u_x1 <= x2 <= u_x2) and (u_y1 <= y2 <= u_y2):
                            cv2.putText(frame, 'Person on UPPER RACK', (x1, y2+30), font, 1, b_c, 1)
                            print('person on upper rack')

                        # lower rack  
                        if (l_x1 <= x2 <= l_x2) and (l_y1 <= y2 <= l_y2):
                            cv2.putText(frame, 'Person on LOWER RACK', (x1, y2+30), font, 1, b_c, 1)
                            print('person on lower rack')
                        
                        
                        
                # 작업중 알림 표시
                elif (label == 'Forklift(H)') or (label == 'Forklift(D)'): 
                    count = 1          # person용 프레임 카운트 초기화 
                    rack_count += 1

                    fork = 250
                    if rack_count >= 3:
                        # upper rack
                        if (u_x1 <= x1 + fork <= u_x2) or (u_x1 <= x2 + fork <= u_x2):
                            cv2.putText(frame, 'Forklift working on UPPER RACK', (10, 50), font, 1, w_c, 2)
                            print('Forklift working on UPPER RACK')

                        # lower rack  
                        if (l_x1 <= x1 + fork <= l_x2) or (l_x1 <= x2 + fork <= l_x2):
                            cv2.putText(frame, 'Forklift working on LOWER RACK', (10, 80), font, 1, w_c, 2)
                            print('Forklift working on LOWER RACK')


                else :
                    count = 1

        # # 감지된 객체에 대한 경계 상자 및 레이블 그리기
        # # annotated_frame = np.squeeze(results.render())

        # # 텍스트 오버레이
        cv2.putText(frame, 'Object Detection With YOLOv8', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('TEST WINDOW', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


video.release()
# out.release()     # 새로운 비디오 파일에 프레임 저장
cv2.destroyAllWindows()