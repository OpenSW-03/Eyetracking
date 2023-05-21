import dlib
import cv2 as cv
import numpy as np
from scipy.spatial import distance
from functools import wraps
import time
import win32gui
import win32con

"""
def close():
    #Function to check if eyes are close
    start_time = time.time()"""


def ScreenOFF():
    # Function to turn off the screen
    return win32gui.SendMessage(win32con.HWND_BROADCAST,
                            win32con.WM_SYSCOMMAND, win32con.SC_MONITORPOWER, 2)

def calculate_EAR(eye):
    # Function to calculate the distance of eye
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        time.sleep(0.05)
        """global lastsave
        if time.time() - lastsave > 5:
            lastsave = time.time()
            tmp.count = 0 """
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp

# 눈을 감으면 메시지가 뜸
@counter
def close():
    ScreenOFF()

# range는 끝값이 포함안됨
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

index = ALL

while True:

    ret, img_frame = cap.read()

    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)

    dets = detector(img_gray, 1)

    for face in dets:

        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

        LEFT_EYE = []  # 왼쪽 눈 좌표 리스트
        RIGHT_EYE = []  # 오른쪽 눈 좌표 리스트

        for n in range(36, 42):  # 오른쪽 눈 감지
            x = shape.part(n).x
            y = shape.part(n).y
            LEFT_EYE.append((x, y))  # 오른쪽 눈의 좌표를 저장
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = shape.part(next_point).x
            y2 = shape.part(next_point).y
            cv.line(img_frame, (x, y), (x2, y2), (0, 255, 0), 1)  # 오른쪽 눈의 좌표로 선을 그림(초록색)

        for n in range(42, 48):  # 왼쪽 눈 감지
            x = shape.part(n).x
            y = shape.part(n).y
            RIGHT_EYE.append((x, y))  # 왼쪽 눈의 좌표를 저장
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = shape.part(next_point).x
            y2 = shape.part(next_point).y
            cv.line(img_frame, (x, y), (x2, y2), (0, 255, 0), 1)  # 왼쪽 눈의 좌표로 선을 그림(초록색)

        left_ear = calculate_EAR(LEFT_EYE)  # 왼쪽 눈 거리 계산
        right_ear = calculate_EAR(RIGHT_EYE)  # 오른쪽 눈 거리 계산

        EAR = (left_ear + right_ear) / 2  # 양쪽 눈 거리 평균 계산
        EAR = round(EAR, 2)  # 소수점 2자리까지 반올림

        if EAR < 0.19:  # 눈 감지 상태인 경우  # 눈 감지 동작 수행
            print(f'close count: {close.count}')  # 눈 감지 횟수 출력
            if close.count == 10:  # 눈 감지 횟수가 10일 경우
                close()
            close.count+=1
        print(EAR)


    cv.imshow('result', img_frame)

    key = cv.waitKey(1)

    if key == 27:
        break

cap.release()
