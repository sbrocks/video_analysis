import numpy as np
import cv2
from keras.preprocessing import image

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture("/floyd/input/dataset/shrofile.mp4")

""" We use cv2.VideoWriter() to save the video """
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('emotion_shrofile.avi',fourcc,20.0,(640,480))



#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

#-----------------------------
emotion_angry = 0
emotion_disgust = 0
emotion_fear = 0
emotion_happy = 0
emotion_sad = 0
emotion_surprise = 0
emotion_neutral = 0
total_frame = 0

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while(cap.isOpened()):
	ret, img = cap.read()
	if ret == True:
		#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		#print(faces) #locations of detected faces

		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
			
			predictions = model.predict(img_pixels) #store probabilities of 7 expressions
			
			#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions[0])
			
			emotion = emotions[max_index]
			
			#write emotion text above rectangle
			cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			if emotion == "angry":
				emotion_angry+=1
			elif emotion == "sad":
				emotion_sad+=1
			elif emotion == "neutral":
				emotion_neutral+=1
			elif emotion == "surprise":
				emotion_surprise+=1
			elif emotion == "happy":
				emotion_happy+=1
			elif emotion == "fear":
				emotion_fear+=1
			elif emotion == "disgust":
				emotion_disgust+=1
			else:
				pass
			#process on detected face end
			#-------------------------

		out.write(img)
		cv2.imshow('img',img)
		total_frame+=1
		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break

	else:
		break

# Percentage calculations
angry_percentage = (emotion_angry/total_frame)*100
sad_percentage = (emotion_sad/total_frame)*100
happy_percentage = (emotion_happy/total_frame)*100
surprise_percentage = (emotion_surprise/total_frame)*100
disgust_percentage = (emotion_disgust/total_frame)*100
fear_percentage = (emotion_fear/total_frame)*100
neutral_percentage = (emotion_neutral/total_frame)*100  

print("Angry: "+str(angry_percentage))
print("Sad: "+str(sad_percentage))
print("Happy: "+str(happy_percentage))
print("Surprise: "+str(surprise_percentage))
print("Disgust: "+str(disgust_percentage))
print("Fear: "+str(fear_percentage))
print("Neutral: "+str(neutral_percentage))

# Release the webcam
cap.release()
out.release()
cv2.destroyAllWindows()