from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import sys
import cv2 as cv
import numpy as np
import os
import random
import threading
import time

class Timer:
    def __init__(self, interval, callback):
        self.interval = interval
        self.callback = callback
        self.is_running = False
        self.thread = None

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run)
            self.thread.start()

    def stop(self):
        self.is_running = False
        self.reset()

    def _run(self):
        count = self.interval
        while count >= 0 and self.is_running:
            self.callback(count)
            time.sleep(0.1)
            count -= 0.1

    def reset(self):
        self.thread = None

class ModelWindow(QWidget): # 모델 표시하는 창을 위해서 사용
    def __init__(self, model_img): # model_img를 받아서 img 표시
        super().__init__()
        self.setWindowTitle("모델 이미지")
        self.setGeometry(200, 200, 400, 400)

        label = QLabel(self) # 매칭 결과 label
        pixmap = QPixmap(model_img)

        self.match_label = QLabel() # 매칭 결과 QLabel
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')
        self.setGeometry(200, 200, 500, 150)

        videoButton = QPushButton('비디오 켜기', self)
        randomImgButton = QPushButton('칠교 이미지 선택', self)
        startButton = QPushButton('Start', self)
        quitButton = QPushButton('나가기', self)

        videoButton.setGeometry(10, 10, 100, 30)
        randomImgButton.setGeometry(120, 10, 150, 30)
        startButton.setGeometry(280, 10, 100, 30)
        quitButton.setGeometry(390, 10, 100, 30)

        videoButton.clicked.connect(self.videoFunction)
        randomImgButton.clicked.connect(self.randomImgFunction)
        startButton.clicked.connect(self.startFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.image_folder_path = "D:/workpy/PJ_tangram/tangramplay"
        self.template_images = []

        self.load_images()

        self.timer = None
        self.model_window = None
        self.sift = cv.xfeatures2d.SIFT_create()
        self.btn_press_count = 0

        self.timer_label = QLabel(self)
        self.timer_label.setGeometry(10, 70, 100, 30)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("background-color: white; border: 1px solid black;")

    def load_images(self):
        for filename in os.listdir(self.image_folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(self.image_folder_path, filename)
                template_img = cv.imread(img_path)
                self.template_images.append((filename, template_img))

    def videoFunction(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            self.close()

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            cv.imshow('video display', self.frame)
            cv.waitKey(1)

    def startFunction(self):
        self.start_timer()

    def start_timer(self):
        if self.timer is not None:
            self.timer.stop()
        self.btn_press_count = 0  # 타이머 시작할 때 버튼 누름 횟수 초기화
        self.timer = Timer(interval=10, callback=self.timer_callback)
        self.timer.start()

    def timer_callback(self, count):
        self.btn_press_count += 1
        self.timer_label.setText(f"{count:.1f} sec")

        if count <= 0.1:
            # 타이머가 0이 되면 매칭 수행
            self.matchFunction()

    def randomImgFunction(self):
        random_image = random.choice(self.template_images)
        self.randomImg = random_image[1]
        cv.imshow('Random Image', self.randomImg)

    def matchFunction(self):
        if not hasattr(self, 'randomImg'):
            print("모델 이미지 선택")
            return
        
        model_img = self.randomImg # 모델 이미지
        gray1=cv.cvtColor(model_img,cv.COLOR_BGR2GRAY)
        kp1, des1 = self.sift.detectAndCompute(gray1, None) # sift로 특징점 추출 이것으로 매칭

        gray2=cv.cvtColor(self.frame,cv.COLOR_BGR2GRAY)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)

        flann_matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

        knn_match=flann_matcher.knnMatch(des1, des2, 2)

        T = 0.8
        good_match = []
        for nearest1, nearest2 in knn_match:
            if(nearest1.distance / nearest2.distance) < T:
                good_match.append(nearest1)

        if len(good_match) > 10: # 매칭된 특징점을 10개 이상인지로 임의로 설정했고 조정해야 할듯
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2) # 소스 이미지 좌표 추출
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)#  대상 이미지 좌표 추출

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0) # Homography행렬 계산 한 이미지의 특정 좌표를 이미지의 좌표로 변환

            h, w = model_img.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            dst = cv.perspectiveTransform(pts, M) # 원근 변환 매트릭스 M사용, 점들의 좌표를 변환하는 함수

            self.frame = cv.polylines(self.frame, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA) # 변환된 모델 이미지 그림
            cv.imshow("display", self.frame)
            cv.waitKey(1)

            match_ratio = len(good_match) / len(knn_match) # 특징점 비율 계산
            accuracy = match_ratio * 100
            if accuracy >= 70:     
                print(f"정답 : {accuracy} %")
            else:
                print(f"오답 : {accuracy} %")
                
        # 임시로 매칭 어떻게 됐나 확인하려고 넣음        
        img_match = np.empty(
            (max(self.randomImg.shape[0], self.frame.shape[0]), self.randomImg.shape[1] + self.frame.shape[1], 3),
            dtype=np.uint8)
        cv.drawMatches(self.randomImg, kp1, self.frame, kp2, good_match, img_match,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("Matching Result", img_match)
        cv.waitKey(0)
        
    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        if self.timer is not None:
            self.timer.stop()
        self.close()

class ResultWindow(QWidget):
    def __init__(self, image, title):
        super().__init__()

        self.setWindowTitle(title)
        self.setGeometry(200, 200, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(10, 10, 780, 480)
        q_image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

        quit_button = QPushButton('나가기', self)
        quit_button.setGeometry(10, 500, 100, 30)
        quit_button.clicked.connect(self.quit_function)

    def quit_function(self):
        self.close()

app = QApplication(sys.argv)
win = Video()
win.show()
app.exec_()