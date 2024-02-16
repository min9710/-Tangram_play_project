from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5 import uic
import sys
import os
import numpy as np
import threading
import time
import cv2
import random


form_class = uic.loadUiType("D:/code/python/Tangram/-Tangram_play_project/Timer/Tangram.ui")[0]
img_dir = 'D:/code/python/Tangram/-Tangram_play_project/dataset/tangramplay/'



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
        if self.thread is not None:
            self.thread.join()  # 타이머 스레드가 완료될 때까지 대기
        self.reset()

    def _run(self):
        count = self.interval
        while count >= 0 and self.is_running:
            self.callback(count)
            time.sleep(0.1)
            count -= 0.1
    
    def reset(self):
        self.thread = None


class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        
        #define function
        self.btn_exit.clicked.connect(self.exitBtnFuction)
        self.btn_start.clicked.connect(self.startBtnFuction)
        
        #counter
        self.label = QLabel()
        self.lb_counter.setAlignment(Qt.AlignCenter)
        
        #timer
        self.timer = None
        self.btn_press_count = 0
        
        
    #function
    def exitBtnFuction(self):
        print("exit clicked")
        
        #윤곽선 검출
        self.find_contours()
    
    def startBtnFuction(self):      
        image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random_image_file = random.choice(image_files)
        
        qPixmapVar = QPixmap(random_image_file)
        self.picTengram.setPixmap(qPixmapVar)
        self.picTengram.setAlignment(Qt.AlignCenter)
        
        if self.timer is not None:
            self.timer.stop()
        self.timer = Timer(interval=10, callback=self.countdown_callback)
        self.timer.start()
        
    def countdown_callback(self, count):
        self.lb_counter.setText("{:.1f}".format(count))
    
    def find_contours(self):
        self.new_window = QMainWindow()  # 새 창 생성
        self.new_window.setWindowTitle("Contours")  # 새 창 제목 설정
        widget = QWidget(self.new_window)
        self.new_window.setCentralWidget(widget)
        
        layout = QVBoxLayout(widget)
        layout.addWidget(self.label)
        
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            cv2.imshow("Contours", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow("Contours") # 카메라 창만 닫기
                break
                
            #qApp.processEvents()  # Qt 이벤트 루프 처리
        
    
if __name__ == "__main__" :
    app = QApplication(sys.argv) 

    myWindow = WindowClass() 
    myWindow.show()

    #app.exec_()
    sys.exit(app.exec_())
#commit test