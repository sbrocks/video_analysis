import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

"""
class TooManyFaces(Exception):
	pass

class NoFaces(Exception):
	pass
"""

def get_landmarks(im):
	rects = detector(im,1)
	#print(len(rects))
	if len(rects)>1:
		print("More than 1 faces")
		return "error"
	if len(rects)==0:
		print("No faces")
		return "error"
	return np.matrix([[p.x,p.y] for p in predictor(im,rects[0]).parts()])

def annotate_landmarks(im,landmarks):
	im = im.copy()
	for idx,point in enumerate(landmarks):
		pos = (point[0,0],point[0,1])
		#cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0,0,255))
		cv2.circle(im,pos,3,color=(255,255,102))
	
	
	#print(landmarks[30],"	",landmarks[8],"	",landmarks[45],"	",landmarks[36],"	",landmarks[64],"	",landmarks[48])
	return im 


# This function will take landmarks as input and compute EAR (Eye Aspect Ratio)
def left_eye(landmarks):
	# 42 to 47
	features = []
	features.append(landmarks[42])
	features.append(landmarks[43])
	features.append(landmarks[44])
	features.append(landmarks[45])
	features.append(landmarks[46])
	features.append(landmarks[47])		
	features = np.squeeze(np.asarray(features))
	l_A = dist.euclidean(features[1],features[5])

	l_B = dist.euclidean(features[2],features[4])
	l_C = dist.euclidean(features[0],features[3])

	l_ear = (l_A+l_B)/(2.0*l_C)
	return l_ear


def right_eye(landmarks):
	# 36 to 41
	right_features = []
	right_features.append(landmarks[36])
	right_features.append(landmarks[37])
	right_features.append(landmarks[38])
	right_features.append(landmarks[39])
	right_features.append(landmarks[40])
	right_features.append(landmarks[41])		
	right_features = np.squeeze(np.asarray(right_features))
	r_A = dist.euclidean(right_features[1],right_features[5])
	r_B = dist.euclidean(right_features[2],right_features[4])
	r_C = dist.euclidean(right_features[0],right_features[3])

	r_ear = (r_A+r_B)/(2.0*r_C)
	return r_ear


def eye_open(image):
	print("1")
	landmarks = get_landmarks(image)

	if landmarks == "error":
		return image,0,0

	image_with_landmarks = annotate_landmarks(image,landmarks)
	left_ear = left_eye(landmarks)
	#print(top_lip_center)
	right_ear = right_eye(landmarks)
	print(left_ear)
	print(right_ear)

	return image_with_landmarks, left_ear, right_ear



# Open the webcam
cap = cv2.VideoCapture(0)
blinks = 0
blink_status = False
i=0
while True:
	ret, frame = cap.read()
	image_landmarks, left_ear, right_ear = eye_open(frame)

	prev_blink_status = blink_status
	
	ear = (left_ear+right_ear)/2.0

	if ear < 0.3:
		blink_status = True

		cv2.putText(frame,"Person is blinking",(50,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
		#cv2.imwrite("frame"+str(i)+".jpg",frame)
		output_text = "Blink Count: " + str(blinks+1)
		i=i+1
		cv2.putText(frame,output_text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)
	else:
		blink_status = False

	if prev_blink_status == True and blink_status == False:
		blinks=blinks+1

	cv2.imwrite("frame"+str(i)+".jpg",frame)
	
	cv2.imshow("Live Landmarks",image_landmarks)
	cv2.imshow("Blink Detection",frame)

	# 13 is the Enter key
	if cv2.waitKey(1) == 13:
		break

cap.release()
cv2.destroyAllWindows()


