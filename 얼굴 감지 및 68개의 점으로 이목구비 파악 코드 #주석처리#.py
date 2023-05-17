import dlib
import cv2 as cv
import numpy as np

detector = dlib.get_frontal_face_detector() # dlib의 얼굴 탐지기 생성

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 얼굴 랜드마크 예측기 생성

cap = cv.VideoCapture(0) # 카메라 열기

# 얼굴 랜드마크 인덱스
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

    ret, img_frame = cap.read() # 카메라에서 프레임 읽기

    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY) # 그레이스케일 변환

    dets = detector(img_gray, 1) # 얼굴 탐지

    for face in dets: # 얼굴 탐지기(dlib.get_frontal_face_detector)로 찾은 얼굴 영역들에 대해 반복

        # 예측기(dlib.shape_predictor)를 사용하여 얼굴 영역(face)에서 68개의 랜드마크를 예측
        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기

        list_points = [] # 예측된 랜드마크 좌표를 저장하기 위한 빈 리스트를 생성
        for p in shape.parts(): # 예측된 랜드마크의 각 좌표에 대해 반복
            list_points.append([p.x, p.y]) # 현재 랜드마크 좌표(p.x, p.y)를 리스트에 추가

        list_points = np.array(list_points) # 좌표 리스트를 NumPy 배열로 변환

        for i, pt in enumerate(list_points[index]): # 선택한 랜드마크 인덱스(index)에 대해 반복
            pt_pos = (pt[0], pt[1]) # 현재 랜드마크 좌표(pt[0], pt[1])를 튜플 형태로 저장
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1) # 랜드마크 좌표에 반지름 2의 초록색 원 그리기
            # 위를 통해 얼굴 이미지에서 랜드마크를 시각적으로 표시

        cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
                     (0, 0, 255), 3) # 얼굴 영역에 사각형 그리기

    cv.imshow('result', img_frame) # 결과 프레임 출력

    key = cv.waitKey(1)

    if key == 27: # ESC 키를 누르면 종료
        break

    elif key == ord('1'): # 1을 누르면 모든 랜드마크 표시
        index = ALL
    elif key == ord('2'): # 2를 누르면 눈썹 랜드마크 표시
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'): # 3을 누르면 눈 랜드마크 표시
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'): # 4를 누르면 코 랜드마크 표시
        index = NOSE
    elif key == ord('5'): # 5를 누르면 입 랜드마크 표시
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE

cap.release()
