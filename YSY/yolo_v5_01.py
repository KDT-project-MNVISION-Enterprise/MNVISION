import torch, cv2

# 모델 및 사전 훈련 가중치 로드
m_file = r"C:\Users\kdp\Desktop\Model_v5.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=m_file) 

# ==========================================================
# 영상 predict 
# ==========================================================

# 영상 열기
f_dir = r"C:\Users\kdp\Desktop\test_video_04.mp4"
cap = cv2.VideoCapture(f_dir)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 객체 탐지
    results = model(frame)
    
    # 결과 시각화
    annotated_frame = results.pandas().xyxy[0].plot(line_width=3)
    cv2.imshow('YOLO_v5', annotated_frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ==========================================================
# 이미지 predict 
# ==========================================================
IMAGE = False

if IMAGE : 
    # 이미지 로드
    img = r"C:\Users\kdp\Desktop\sample_01.jpg"  # 예측을 수행할 이미지 경로

    # 예측 수행
    results = model(img)

    # 결과 출력
    results.print()  # 결과 출력
    results.show()   # 결과 이미지 시각화
    # results.save()   # 결과 이미지 저장

    # 결과를 DataFrame으로 변환하여 필요 시 활용
    df = results.pandas().xyxy[0]
    print(df)
