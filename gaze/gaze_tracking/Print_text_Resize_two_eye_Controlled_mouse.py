import cv2
import mediapipe as mp
import pyautogui
from gaze_tracking import GazeTracking

gaze = GazeTracking() # 오픈소스 gazetracking 사용

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 500) # 카메라 화면 창의 가로 길이를 500으로 설정
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 500) # 카메라 화면 창의 세로 길이를 500으로 설정

window_name = 'Eye Controlled Mouse' # 윈도우 창 이름
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, 0, 0) # 카메라를 좌상단에 고정

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cam.read()

    # gaze_tracking에 분석할 프레임을 전달한다
    gaze.refresh(frame) # gaze.refresh에서 cvtColor을 회색으로 바꾼다
    frame = gaze.annotated_frame()

    # gaze_tracking에서 변경한 값을 다시 변경
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

        screen_x = screen_w * ((left_eye[0].x + right_eye[0].x) / 2)
        screen_y = screen_h * ((left_eye[0].y + right_eye[0].y) / 2)
        pyautogui.moveTo(screen_x, screen_y) # 마우스를 움직인다

    if gaze.is_blinking(): # 눈을 깜빡이면
        # 눈을 깜빡이면, 카메라 화면 왼쪽 상단에 Blinking 출력
        cv2.putText(frame, "Blinking", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (147, 58, 31), 2)
        # cv2.putText(이미지/영상, 작성할_문자열, 문자열_좌표, 폰트, 문자열_크기, 문자열_색상(r,g,b), 문자열_굵기)

        pyautogui.click() # 마우스 클릭
        pyautogui.sleep(1)

    # 왼쪽눈과 오른쪽 눈의 좌표를 카메라 화면에 출력한다
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil: " + str(left_pupil), (10, 440), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (10, 470), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)

    cv2.imshow(window_name, frame)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # 항상 윈도우 창을 최상단에 표시

    # Esc 키를 누르면 프로그램 종료
    if cv2.waitKey(1) == 27:
        break

cam.release() # 웹캠과의 연결을 끊는다
cv2.destroyAllWindows() # 웹캠 영상을 보여주기 위해 생성했던 창을 없앤다
