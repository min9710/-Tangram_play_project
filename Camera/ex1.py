from PyQt5.QtWidgets import *
import sys
import cv2 as cv
import numpy as np
       
class Video(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')	# 윈도우 이름과 위치 지정
        self.setGeometry(200,200,500,100)

        videoButton=QPushButton('비디오 켜기',self)	# 버튼 생성
        captureButton=QPushButton('프레임 잡기',self)
        saveButton=QPushButton('프레임 저장',self)
        quitButton=QPushButton('나가기',self)
        
        videoButton.setGeometry(10,10,100,30)		# 버튼 위치와 크기 지정
        captureButton.setGeometry(110,10,100,30)
        saveButton.setGeometry(210,10,100,30)
        quitButton.setGeometry(310,10,100,30)
        
        videoButton.clicked.connect(self.videoFunction) # 콜백 함수 지정
        captureButton.clicked.connect(self.captureFunction)         
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)
       
    def videoFunction(self):
        self.cap=cv.VideoCapture(0,cv.CAP_DSHOW)	# 카메라와 연결 시도
        if not self.cap.isOpened(): self.close()
            
        while True:
            ret,self.frame=self.cap.read() 
            if not ret: break            
            cv.imshow('video display',self.frame)
            cv.waitKey(1)
        
    def captureFunction(self): # 프레임 잡기
        self.capturedFrame=self.frame
        cv.imshow('Captured Frame',self.capturedFrame)
        
    def saveFunction(self):				# 파일 저장
        img1=cv.imread('a.png') # 버스를 크롭하여 모델 영상으로 사용
        gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
        img2=self.capturedFrame			     # 장면 영상
        gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

        sift=cv.SIFT_create()
        kp1,des1=sift.detectAndCompute(gray1,None)
        kp2,des2=sift.detectAndCompute(gray2,None)
        print('특징점 개수:',len(kp1),len(kp2)) 

        flann_matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_match=flann_matcher.knnMatch(des1,des2,2)

        T=0.7
        good_match=[]
        for nearest1,nearest2 in knn_match:
            if (nearest1.distance/nearest2.distance)<T:
                good_match.append(nearest1)

        img_match=np.empty((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3),dtype=np.uint8)
        cv.drawMatches(img1,kp1,img2,kp2,good_match,img_match,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imshow('Good Matches', img_match)
        
    def quitFunction(self):
        self.cap.release()				# 카메라와 연결을 끊음
        cv.destroyAllWindows()
        self.close()
                
app=QApplication(sys.argv) 
win=Video() 
win.show()
app.exec_()
