import cv2
import mediapipe as mp
import pyautogui
from gaze_tracking import GazeTracking
from time import time
import win32gui
import win32con

gaze = GazeTracking() # 오픈소스 gazetracking 사용

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 500) # 카메라 화면 창의 가로 길이를 500으로 설정
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 500) # 카메라 화면 창의 세로 길이를 500으로 설정

window_name = 'Eye Controlled Mouse' # 윈도우 창 이름
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, 0, 0) # 카메라를 좌상단에 고정

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
pyautogui.moveTo(screen_w // 2, screen_h // 2)  # 초기 마우스 위치를 화면 중앙으로 이동

start_blinking_time = 0
blinking_time = 0
is_clicked = False
start_count = 0
end_count = 0

def move_mouse(x, y):
    # 마우스를 지정된 좌표로 이동시키는 함수
    inverted_x = screen_w - x  # 좌우 반전된 좌표 계산
    pyautogui.moveTo(inverted_x, y)

def ScreenOFF():
    # Function to turn off the screen
    return win32gui.SendMessage(win32con.HWND_BROADCAST,
                            win32con.WM_SYSCOMMAND, win32con.SC_MONITORPOWER, 2)

def close():
    ScreenOFF()

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


    # 마우스 커서 이동
    left_coords = gaze.pupil_left_coords() # 왼쪽 눈 좌표
    right_coords = gaze.pupil_right_coords() # 오른쪽 눈 좌표

    if left_coords is not None and right_coords is not None: # None일 때 제외
        left_x, left_y = left_coords
        right_x, right_y = right_coords

        screen_x = (left_x + right_x) / 2
        screen_y = (left_y + right_y) / 2

        move_mouse(int(screen_x), int(screen_y))  # 수정된 부분


    if gaze.is_blinking():  # 눈을 감을 때
        if start_blinking_time == 0: # 눈감고 있는 시간을 안재고 있었으면
            start_blinking_time = time()  # 눈감고 있는 시간 시작

        else:  # 눈을 감고 있는 시간 재고 있으면
            # 눈을 깜빡이면, 카메라 화면 왼쪽 상단에 Blinking 출력
            cv2.putText(frame, "Blinking", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (147, 58, 31), 2)
            # cv2.putText(이미지/영상, 작성할_문자열, 문자열_좌표, 폰트, 문자열_크기, 문자열_색상(r,g,b), 문자열_굵기)

            blinking_time = time() - start_blinking_time
            # 눈을 감은지 3초가 초과하면
            if blinking_time > 3.0:
                start_count = time()
                while end_count < 3.0:
                    end_count = time() - start_count
                    cv2.putText(frame, "You've been closing your eyes for 3 seconds.", (50, 230), cv2.FONT_HERSHEY_DUPLEX, 0.7, (30, 17, 155), 2)
                    cv2.putText(frame, "Shut down the system.", (170, 260), cv2.FONT_HERSHEY_DUPLEX, 0.7, (30, 17, 155), 2)
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1)

                start_blinking_time = 0
                blinking_time = 0
                ScreenOFF()
                exit()

            # 눈을 감았다 뜨면(깜빡이면)
            if (left_eye[0].y - left_eye[1].y) > 0.026 and (right_eye[0].y - right_eye[1].y) > 0.026:
                while True:
                    if not is_clicked:  # 연속 클릭 방지
                        pyautogui.click()  # 마우스 클릭
                        is_clicked = True
                        start_blinking_time = 0
                        blinking_time = 0
                        break

        is_clicked = False

    else:
        start_blinking_time = 0
        blinking_time = 0
        is_clicked = False

    cv2.putText(frame, "Blinking time: {:.4f}".format(blinking_time), (10, 410), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)

    # 눈동자의 방향을 출력한다
    if gaze.is_right():
        cv2.putText(frame, "Looking right", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)
    elif gaze.is_left():
        cv2.putText(frame, "Looking left", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)
    elif gaze.is_center():
        cv2.putText(frame, "Looking center", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)

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