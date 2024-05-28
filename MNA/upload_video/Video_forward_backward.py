import cv2

def play_video(video_path):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 비디오 프레임 속도와 총 프레임 수 얻기
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video Duration: {duration:.2f} seconds")

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 현재 프레임을 화면에 표시
        cv2.imshow('Video', frame)

        # 10ms마다 키 입력을 기다림
        key = cv2.waitKey(10) & 0xFF

        # 'q' 키를 누르면 종료
        if key == ord('q'):
            break

        # 'f' 키를 누르면 10초 앞으로 건너뜀
        elif key == ord('f'):
            current_frame += int(10 * fps)
            if current_frame >= frame_count:
                current_frame = frame_count - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # 'b' 키를 누르면 10초 뒤로 건너뜀
        elif key == ord('b'):
            current_frame -= int(10 * fps)
            if current_frame < 0:
                current_frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

# 비디오 파일 경로를 지정하여 재생
video_path = 'upload_video/Long.avi'
play_video(video_path)
