from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
import cv2 as cv
import numpy as np
import random
import os
import sys
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

class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tangramplay")
        self.setGeometry(200, 300, 550, 450)

        self.videoButton = QPushButton("ON", self)    
        self.captureButton = QPushButton("모델", self)
        self.matchButton = QPushButton("매칭", self)
        self.quitButton = QPushButton("나가기", self)
        
        self.videoButton.setGeometry(110, 10, 100, 30)      
        self.captureButton.setGeometry(10, 10, 100, 30)
        self.matchButton.setGeometry(210, 10, 100, 30)
        self.quitButton.setGeometry(310, 10, 100, 30)
        
        self.videoButton.clicked.connect(self.videoFunction) 
        self.captureButton.clicked.connect(self.captureFunction)         
        self.matchButton.clicked.connect(self.startFunction)
        self.quitButton.clicked.connect(self.quitFunction)
        
        self.image_folder_path = "D:/python_workspace/tangramplay"
        self.load_img()

        self.timer_label = QLabel(self)
        self.timer_label.setGeometry(10, 70, 100, 30)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("background-color: white; border: 1px solid black;")

        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_callback)
        self.time_left = 5
        self.btn_press_count = 0
        self.sift = cv.xfeatures2d.SIFT_create()
        self.model_img = None
        self.frame = None

        self.model_label = QLabel(self)
        self.model_label.setGeometry(170, 70, 370, 300)
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setStyleSheet("border: 1px solid black;")

    def load_img(self): 
        self.template_img = []
        for filename in os.listdir(self.image_folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(self.image_folder_path, filename)
                self.template_img.append(img_path)

    def captureFunction(self): 
        random_img = random.choice(self.template_img)
        self.model_img = cv.imread(random_img)
        pixmap = QPixmap(random_img)
        self.model_label.setPixmap(pixmap)

    def videoFunction(self): 
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            self.close()         
        while True:
            ret, self.frame = self.cap.read() 
            if not ret:
                break            
            cv.imshow("display", self.frame)
            cv.waitKey(1)

    def startFunction(self):
        self.start_timer()

    def start_timer(self):
        if self.timer is not None:
            self.timer.stop()
        self.btn_press_count = 0  
        self.timer = Timer(interval=5, callback=self.timer_callback)  # 5초로 설정
        self.timer.start()

    def timer_callback(self, count):
        self.btn_press_count += 1
        self.timer_label.setText(f"{count:.1f} sec")

        if count <= 0.1:
            self.matchFunction()

    def randomImgFunction(self):
        random_image = random.choice(self.template_img)
        self.model_img = cv.imread(random_image)
        cv.imshow('Random Image', self.model_img)

    def matchFunction(self):
        if self.model_img is None:
            print("모델 이미지 선택")
            return
        
        gray1 = cv.cvtColor(self.model_img, cv.COLOR_BGR2GRAY)
        kp1, des1 = self.sift.detectAndCompute(gray1, None)  

        gray2 = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)

        flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_match = flann_matcher.knnMatch(des1, des2, 2)

        T = 0.8
        good_match = []
        for nearest1, nearest2 in knn_match:
            if(nearest1.distance / nearest2.distance) < T:
                good_match.append(nearest1)

        if len(good_match) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            if M is not None:
                h, w = self.model_img.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                self.frame = cv.polylines(self.frame, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
                cv.imshow("display", self.frame)
                cv.waitKey(1)

                match_ratio = len(good_match) / len(knn_match)
                accuracy = match_ratio * 100
                if accuracy >= 70:
                    print(f"정답 : {accuracy : .2f} %")
                else:
                    print(f"오답 : {accuracy: .2f} %")
            else:
                print("Homography 행렬 형태 올바르지 않음")

    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        if self.timer is not None:
            self.timer.stop()
        self.close()

app = QApplication(sys.argv)
win = Video()
win.show()
app.exec_()
