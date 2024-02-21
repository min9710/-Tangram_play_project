from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import sys
import os
import numpy as np
import cv2
import random

form_class = uic.loadUiType('D:/code/python/Tangram/tangram/Timer/Tangram__.ui')[0]
img_dir = 'D:/code/python/Tangram/tangram/imgs/'

#cameraLb 연결
class CameraThread(QThread):
    frameCaptured = pyqtSignal(QImage)
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(0)
        self.is_running = False
        self.main_window = main_window

    def start(self):
        self.is_running = True
        super().start()
    
    def run(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # 케니 엣지 검출 적용
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(frame_gray, 100, 200)

                # OpenCV 배열을 QImage로 변환
                q_img = QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8)
                self.frameCaptured.emit(q_img)
            else:
                print("Failed to capture frame from camera. Skipping empty frame.")

    def stop(self):
        self.is_running = False
        self.wait()

    def convert_cvimage_to_qimage(self, cv_image):
        h, w = cv_image.shape
        q_img = QImage(cv_image.data, w, h, w, QImage.Format_Grayscale8)
        return q_img    

class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Tengram")

        #btn
        self.exitBtn.clicked.connect(self.exitBtnFunction)
        self.startBtn.clicked.connect(self.startBtnFunction)

        #camera
        self.camera_thread = CameraThread(main_window=self)  # CameraThread에 메인 윈도우 객체 전달
        self.camera_thread.frameCaptured.connect(self.displayFrame)
        self.camera_thread.frameCaptured.connect(self.process_frame_qimage)
        
        #timer
        self.countdownTimer = QTimer(self)
        self.countdownTimer.timeout.connect(self.updateCountdown)
        self.countdownDuration = 10.0
        self.currentCountdown = self.countdownDuration

        #label setting
        self.label = QLabel()
        self.cntLb.setAlignment(Qt.AlignCenter)

        self.tempImg = None

        # Canny 엣지 검출 모드 설정
        self.canny_mode = False

    #---------camera, counter callback---------
    def start_camera(self):
        if not self.camera_thread.isRunning():
            self.camera_thread.start()

    def stop_camera(self):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            print("Camera stopped.")

    def updateCountdown(self):
        self.currentCountdown -= 0.1
        self.cntLb.setText(f"{self.currentCountdown:.1f}s")
        print(f"Countdown: {self.currentCountdown:.1f}s")
        if self.currentCountdown <= 0:
            self.countdownTimer.stop()
            self.cntLb.setText("Done!")
            print("Countdown finished!")
            self.stop_camera()


    #---------ssab eva in bubun-----------
    def process_frame_qimage(self, qimage):
        cv_image = self.convert_qimage_to_cvimage(qimage)
        self.process_frame(cv_image)

    def process_frame(self, frame):
        if self.tempImg is not None:
            # 케니 엣지 검출 적용
            edges = cv2.Canny(frame, 100, 200)

            # 템플릿 이미지도 그레이스케일로 변환
            tempImg_gray = cv2.cvtColor(self.tempImg, cv2.COLOR_BGR2GRAY)

            # 템플릿 매칭 수행
            result = cv2.matchTemplate(edges, tempImg_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 정확도 계산 및 표시
            accuracy = max_val * 100
            self.accuracyLb.setText(f"Accuracy: {accuracy:.2f}%")

            # 매칭된 위치에 사각형 그리기
            w, h = tempImg_gray.shape[::-1]  # 템플릿 이미지의 너비와 높이
            top_left = max_loc  # 매칭된 영역의 좌상단 좌표
            bottom_right = (top_left[0] + w, top_left[1] + h)  # 매칭된 영역의 우하단 좌표
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)  # 빨간색 사각형으로 매칭 영역 표시

            # 프레임을 RGB로 변환하고 QImage로 변환하여 표시
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.displayFrame(convert_to_qt_format)




    def displayFrame(self, image):
        self.cameraLb.setPixmap(QPixmap.fromImage(image).scaled(
            self.cameraLb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def capture_and_match(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)

    def stopCapturing(self):
        self.camera_thread.stop()
        self.countdownTimer.stop()
        self.cntLb.setText("")


    #---------image format-----------    
    def convert_qimage_to_cvimage(self, qimage):
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        size = qimage.size()
        if qimage.format() == QImage.Format_RGB888: # RGB 포맷일 경우
            mat = np.array(ptr).reshape(size.height(), size.width(), 3)
        elif qimage.format() in [QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied]: # ARGB 포맷일 경우
            mat = np.array(ptr).reshape(size.height(), size.width(), 4)
        elif qimage.format() == QImage.Format_Grayscale8: # 그레이스케일 포맷일 경우
            mat = np.array(ptr).reshape(size.height(), size.width())
        else: # 다른 포맷을 사용하는 경우, 추가적인 처리가 필요할 수 있음
            raise ValueError("Unsupported QImage format!")

        # QImage 포맷에 따라 BGR로 변환 (OpenCV에서 사용하기 위함)
        if qimage.format() in [QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied, QImage.Format_RGB888]:
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

        return mat


    #---------------btn-------------
    def startBtnFunction(self):
        # 이미지 디렉토리에서 이미지 파일 목록 가져오기
        image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random_image_file = random.choice(image_files) # 랜덤 이미지 선택
        self.tempImg = cv2.imread(random_image_file) # 선택된 이미지를 tempImg에 로드 및 할당

        qPixmapVar = QPixmap(random_image_file)
        self.picTengram.setPixmap(qPixmapVar)
        self.picTengram.setAlignment(Qt.AlignCenter)

        # 카메라 쓰레드 시작
        if not self.camera_thread.isRunning():
            self.camera_thread.start()
        else:
            print("Camera is already running.")

        # 카운트다운 시작
        if self.countdownTimer.isActive():
            self.countdownTimer.stop()
        self.currentCountdown = self.countdownDuration
        self.cntLb.setText(f"{self.currentCountdown:.1f}s")
        self.countdownTimer.start(100)


    def exitBtnFunction(self):
        self.stopCapturing()
        self.countdownTimer.stop()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
