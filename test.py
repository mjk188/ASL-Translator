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

cv2.namedWindow('trackbar')
cv2.createTrackbar('Y_min','trackbar',0,255,nothing)
cv2.createTrackbar('Y_max','trackbar',0,255,nothing)
cv2.createTrackbar('Cr_min','trackbar',0,255,nothing)
cv2.createTrackbar('Cr_max','trackbar',0,255,nothing)
cv2.createTrackbar('Cb_min','trackbar',0,255,nothing)
cv2.createTrackbar('Cb_max','trackbar',0,255,nothing)
while(cap.isOpened()):
	Y_min = cv2.getTrackbarPos('Y_min','trackbar')
	Y_max = cv2.getTrackbarPos('Y_max','trackbar')
	Cr_min = cv2.getTrackbarPos('Cr_min','trackbar')
	Cr_max = cv2.getTrackbarPos('Cr_max','trackbar')
	Cb_min = cv2.getTrackbarPos('Cb_min','trackbar')
	Cb_max = cv2.getTrackbarPos('Cb_max','trackbar')
	_,img=cap.read()
	cv2.rectangle(img,(350,128),(600,400),(255,0,0),3)
	img1=img[128:400,350:600]
	img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
	blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
	skin_ycrcb_min = np.array((Y_min,Cr_min,Cb_min))
	skin_ycrcb_max = np.array((Y_max,Cr_max,Cb_max))
	mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)
	#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#ret,mask = cv2.threshold(gray.copy(),20,255,cv2.THRESH_BINARY)
	contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt=getMaxContour(contours,40)
	if cnt!=None:
		x,y,w,h = cv2.boundingRect(cnt)
		imgT=img1[y:y+h,x:x+w]
		imgT=cv2.bitwise_and(imgT,imgT,mask=mask[y:y+h,x:x+w])
		imgT=cv2.resize(imgT,(300,300))
		cv2.imshow('Trainer',imgT)
	cv2.imshow('Frame',img)
	cv2.imshow('Thresh',mask)
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