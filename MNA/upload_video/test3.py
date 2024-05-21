import cv2

# 동영상 파일 경로
video_path = "Streaming/person.avi"

# 동영상을 읽어올 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 동영상이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 동영상 프레임 너비와 높이
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 윈도우 생성 및 윈도우 이름 설정
cv2.namedWindow("Video Player")

# 동영상 재생 제어 변수
is_playing = True

while True:
    # 동영상 재생 중일 때 프레임을 읽어옴
    if is_playing:
        ret, frame = cap.read()

        # 동영상의 마지막 프레임에 도달하면 종료
        if not ret:
            break

        # 프레임을 윈도우에 표시
        cv2.imshow("Video Player", frame)

    # 키 입력 대기 (30ms)
    key = cv2.waitKey(30)

    # 'q' 키를 누르면 종료
    if key == ord('q'):
        break
    # 'p' 키를 누르면 재생/일시정지 토글
    elif key == ord('p'):
        is_playing = not is_playing

# 캡처 객체 및 윈도우 제거
cap.release()
cv2.destroyAllWindows()
