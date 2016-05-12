import cv2
import numpy as np


cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(cam)
i=1
j=1
name=""

def nothing(x) :
    pass

#Get the biggest Controur
def getMaxContour(contours,minArea=200):
    maxC=None
    maxArea=minArea
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>maxArea):
            maxArea=area
            maxC=cnt
    return maxC


while(cap.isOpened()):
	_,img=cap.read()
	cv2.rectangle(img,(350,128),(600,400),(255,0,0),3)
	img1=img[128:400,350:600]
	cv2.line(img,(350,128),(600,400),(255,0,0),5)
	cv2.imshow('Frame',img)
	cv2.imshow('Thresh',img1)
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	if k == 13:
		name=str(i)+"_"+str(j)+".jpg"
		cv2.imwrite(name,imgT)
		if(j<20):
			j+=1
		else:
			while(0xFF & cv2.waitKey(0)!=ord('n')):
				j=1
			j=1
			i+=1
		

cap.release()        
cv2.destroyAllWindows()