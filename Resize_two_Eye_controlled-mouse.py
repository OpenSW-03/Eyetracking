import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 300) # 카메라 창의 가로 길이를 300으로 설정
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 300) # 카메라 창의 세로 길이를 300으로 설정


window_name = 'Eye Controlled Mouse'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, 0, 0) # 카메라를 좌상단에 고정

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        left_eye = [landmarks[145], landmarks[159]] # 왼쪽 눈 포인트
        right_eye = [landmarks[374], landmarks[386]] # 오른쪽 눈 포인트

        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

        for landmark in right_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # 왼쪽 눈과 오른쪽 눈이 동시에 깜빡일 때, 마우스 클릭(눈의 높이가 0.008보다 작아질 때)
        if (left_eye[0].y - left_eye[1].y) < 0.008 and (right_eye[0].y - right_eye[1].y) < 0.008:
            pyautogui.click()
            pyautogui.sleep(1)

        screen_x = screen_w * ((left_eye[0].x + right_eye[0].x) / 2)
        screen_y = screen_h * ((left_eye[0].y + right_eye[0].y) / 2)
        pyautogui.moveTo(screen_x, screen_y) # 마우스를 움직인다

    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # 항상 최상단에 표시
    cv2.waitKey(1)
