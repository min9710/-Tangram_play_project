from PyQt5.QtWidgets import *
import sys
import cv2 as cv
import numpy as np
       
class Video(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')
        self.setGeometry(200,200,500,100)

        videoButton=QPushButton('비디오 켜기',self)	
        captureButton=QPushButton('프레임 잡기',self)
        matchButton=QPushButton('매칭',self)
        quitButton=QPushButton('나가기',self)
        
        videoButton.setGeometry(10,10,100,30)		
        captureButton.setGeometry(110,10,100,30)
        matchButton.setGeometry(210,10,100,30)
        quitButton.setGeometry(310,10,100,30)
        
        videoButton.clicked.connect(self.videoFunction) 
        captureButton.clicked.connect(self.captureFunction)         
        matchButton.clicked.connect(self.matchFunction)
        quitButton.clicked.connect(self.quitFunction)
       
    def videoFunction(self):
        self.cap=cv.VideoCapture(0,cv.CAP_DSHOW)
        if not self.cap.isOpened(): self.close()
            
        while True:
            ret,self.frame=self.cap.read() 
            if not ret: break            
            cv.imshow('video display',self.frame)
            cv.waitKey(1)
        
    def captureFunction(self): # 카운트 다운 끝나면 프레임 잡으면 될 듯
        self.capturedFrame=self.frame
        cv.imshow('Captured Frame',self.capturedFrame)
        
    def matchFunction(self):		
        img1=cv.imread('a.png') # 도안
        gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
        img2=self.capturedFrame # 타이머 끝났을 떄 프레임
        gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

        sift=cv.SIFT_create()
        kp1,des1=sift.detectAndCompute(gray1,None)
        kp2,des2=sift.detectAndCompute(gray2,None)

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
        self.cap.release()
        cv.destroyAllWindows()
        self.close()
                
app=QApplication(sys.argv) 
win=Video() 
win.show()
app.exec_()
