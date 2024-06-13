from ultralytics import YOLO

# 일단 고정되어있다고 가정
u_x1, u_y1, u_x2, u_y2 = 866, 294, 1111, 33      # cam 6
l_x1, l_y1, l_x2, l_y2 = 897, 293, 1232, 547

# 저장된 모델 불러오기
m_file = r"C:\Users\kdp\Desktop\240531_best_v8.pt"
model = YOLO(m_file)

# 예측하기 - 사진
file = "C:/Users/kdp/PycharmProjects/My_Project/TEST/try_02/01frame_12582.jpg"
results = model.predict(source = file, save=True, conf=0.7)

# 결과물 확인
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    speeds = result.speed
    obb = result.obb

    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        label = model.names[class_id]
        print('-'*30)
        print(f"좌표: ({x1}, {y1}) - ({x2}, {y2}) \n라벨: {label}")
        print(f"speeds : {speeds} \nobb : {obb}")

        score = result.boxes.conf  # confidence scores
        box_cls = result.boxes.cls
        print(f"score : {score} \nclass : {box_cls}")
        print('-'*30)
