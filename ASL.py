import cv2
import numpy as np
import util as ut
import svm_train as st 
import re
model=st.trainSVM(17)
#create and train SVM model each time coz bug in opencv 3.1.0 svm.load() https://github.com/Itseez/opencv/issues/4969
cam=int(raw_input("Enter Camera number: "))
cap=cv2.VideoCapture(cam)
font = cv2.FONT_HERSHEY_SIMPLEX

def nothing(x) :
    pass

text= " "

temp=0
previouslabel=None
previousText=" "
label = None
while(cap.isOpened()):
	_,img=cap.read()
	cv2.rectangle(img,(900,100),(1300,500),(255,0,0),3) # bounding box which captures ASL sign to be detected by the system
	img1=img[100:500,900:1300]
	img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
	blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
	skin_ycrcb_min = np.array((0, 138, 67))
	skin_ycrcb_max = np.array((255, 173, 133))
	mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection
	contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, 2) 
	cnt=ut.getMaxContour(contours,4000)						  # using contours to capture the skin filtered image of the hand
	if cnt!=None:
		gesture,label=ut.getGestureImg(cnt,img1,mask,model)   # passing the trained model for prediction and fetching the result
		if(label!=None):
			if(temp==0):
				previouslabel=label
	        if previouslabel==label :
	        	previouslabel=label
	        	temp+=1
	        else :
	        	temp=0
	        if(temp==40):
	        	if(label=='P'):
	        	
	        		label=" "
	        	text= text + label
	        	if(label=='Q'):
	        		words = re.split(" +",text)
	        		words.pop()
	        		text = " ".join(words)
	        		#text=previousText
	        	print text
	        	
		cv2.imshow('PredictedGesture',gesture)				  # showing the best match or prediction
		cv2.putText(img,label,(50,150), font,8,(0,125,155),2)  # displaying the predicted letter on the main screen
		cv2.putText(img,text,(50,450), font,3,(0,0,255),2)
	cv2.imshow('Frame',img)
	cv2.imshow('Mask',mask)
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	

cap.release()        
cv2.destroyAllWindows()
