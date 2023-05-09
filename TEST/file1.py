# 패키지 설치
# pip install dlib opencv-python
#
# 학습 모델 다운로드
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
import dlib
import cv2 as cv
import numpy as np

detector = dlib.get_frontal_face_detector() #얼굴 검출을 위해 디폴트 검출기 사용

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #학습모델 로드

cap = cv.VideoCapture(0) #웹캠으로부터 영상 가져와 입력값으로 사용

# range는 끝값이 포함안됨                부위별로 리스트를 정의
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

index = ALL #초기값은 전체 랜드마크를 보여줌

while True: #웹캠으로부터 입력 받기 위해 무한반복함

    ret, img_frame = cap.read() #웹캠에서 이미지를 가져와서

    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY) #그레이 스케일로 변환

    dets = detector(img_gray, 1) #주어진 이미지에서 얼굴 검출, 1은 업샘플링 횟수(이미지 확대)

    for face in dets: #검출된 얼굴 갯수만큼 반복

        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기
                                             #주어진 이미지 img_frame의 검출된 얼굴 영역 face에서 랜드마크 검출
        list_points = []                     #검출된 랜드마크 리스트에 저장
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points) #리스트를 numpy 배열로 변환.

        for i, pt in enumerate(list_points[index]): #검출된 랜드마크 중 index 변수에 지정된 부위만
            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1) #이미지에 원으로 그림.

        cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()), #검출된 얼굴 영역에 빨간색 사각형 그림
                     (0, 0, 255), 3)

    cv.imshow('result', img_frame) #결과 이미지 화면에 띄움

    key = cv.waitKey(1) #키보드 입력받음

    if key == 27: #ESC 키 누르면 종료.
        break

    elif key == ord('1'):                   #입력된 숫자에 따라 index 변수에 앞에서 저장한 리스트 대입
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):                   #3을 누르면 눈에 있는 랜드마크에만 원이 그려짐
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE

cap.release()