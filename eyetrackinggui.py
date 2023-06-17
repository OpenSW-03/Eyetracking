import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import subprocess

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("eyetrackingui.ui")[0]

# 실행할 Python 실행 파일 경로
python_exe = 'C:\PycharmProjects\blink+screenoff+resize\build\trial.2.\trial.2.exe'

# 실행할 Python 파일 경로
python_file = 'C:\PycharmProjects\blink+screenoff+resize\build\trial.2.\trial.2.py'


#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.blink)

    def blink(self):
        # subprocess 모듈을 사용하여 외부 프로세스 실행
        subprocess.run([python_exe, python_file])

if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    print('11')
    app = QApplication(sys.argv)

    #WindowClass의 인스턴스 생성
    print('22')

    myWindow = WindowClass()

    #프로그램 화면을 보여주는 코드
    print('33')

    myWindow.show()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    print('44')

    try:
        sys.exit(app.exec_())
    except Exception as e:
        print("An exception occurred:", str(e))
        sys.exit(1)

    print('55')

    """
    def close():
        #Function to check if eyes are close
        start_time = time.time()"""
