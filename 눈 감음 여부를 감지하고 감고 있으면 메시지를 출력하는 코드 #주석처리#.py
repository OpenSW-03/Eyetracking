import cv2
import dlib
from functools import wraps
from scipy.spatial import distance
import RPi.GPIO as GPIO
import time
from tkinter import *

def calculate_EAR(eye): # 눈 거리 계산 함수 정의
    A = distance.euclidean(eye[1], eye[5]) # 눈의 수직 거리 계산
    B = distance.euclidean(eye[2], eye[4]) # 눈의 수평 거리 계산
    C = distance.euclidean(eye[0], eye[3]) # 눈의 대각선 거리 계산
    ear_aspect_ratio = (A + B) / (2.0 * C) # 눈 거리의 비율 계산
    return ear_aspect_ratio

# 카메라 셋팅
cap = cv2.VideoCapture(0)
cap.set(3, 640) # 프레임 가로 크기 설정
cap.set(4, 480) # 프레임 세로 크기 설정

# dlib 인식 모델 정의
hog_face_detector = dlib.get_frontal_face_detector() # 얼굴 탐지기 모델
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 얼굴 랜드마크 모델

def counter(func): # 함수 호출 횟수를 세는 데코레이터 함수 정의
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        time.sleep(0.05)
        global lastsave
        if time.time() - lastsave > 5:
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp

# 눈을 감으면 메시지가 뜸
@counter
def close(): # 눈 감지 시 동작하는 함수 정의
    window=Tk()
    label=Label(window, text="눈 감음을 감지하였습니다.")
    label.pack()
    window.mainloop()

while True:
    _, frame = cap.read() # 카메라에서 프레임 읽기
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # 그레이스케일 변환

    faces = hog_face_detector(gray) # 얼굴 탐지
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face) # 얼굴 랜드마크 탐지
        leftEye = [] # 왼쪽 눈 좌표 리스트
        rightEye = [] # 오른쪽 눈 좌표 리스트

        for n in range(36,42): # 오른쪽 눈 감지
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	leftEye.append((x,y)) # 오른쪽 눈의 좌표를 저장
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv.line(frame,(x,y),(x2,y2),(0,255,0),1) # 오른쪽 눈의 좌표로 선을 그림(초록색)

        for n in range(42,48): # 왼쪽 눈 감지
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	rightEye.append((x,y)) # 왼쪽 눈의 좌표를 저장
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv.line(frame,(x,y),(x2,y2),(0,255,0),1) # 왼쪽 눈의 좌표로 선을 그림(초록색)

        left_ear = calculate_EAR(leftEye) # 왼쪽 눈 거리 계산
        right_ear = calculate_EAR(rightEye) # 오른쪽 눈 거리 계산

        EAR = (left_ear+right_ear)/2 # 양쪽 눈 거리 평균 계산
        EAR = round(EAR,2) # 소수점 2자리까지 반올림

        if EAR<0.19: # 눈 감지 상태인 경우
            close() # 눈 감지 동작 수행
            print(f'close count: {close.count}') # 눈 감지 횟수 출력
            if close.count == 15: # 눈 감지 횟수가 15일 경우
                print("You are sleeping") # 사용자가 졸고 있음을 출력
        print(EAR)

    cv.imshow("Are you Sleepy", frame)

    key = cv.waitKey(30)
    if key == 27: # ESC 키를 누르면 종료
        break
        
cap.release()
cv.destroyAllWindows()
