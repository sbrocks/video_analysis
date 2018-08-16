# Importing the libraries
import numpy as np
import cv2
from keras.preprocessing import image
import dlib
from imutils import face_utils

#------------------------------------------------------------------------------------------------------
### EMOTION INITIALIZATION PART START -------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# Loading the OpenCV face detection HaarClassifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

""" We use cv2.VideoWriter() to save the video """
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('emotion_shrofile.avi',fourcc,20.0,(640,480))



#-----------------------------
# Loading the facial expression recognition model
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') 

#-----------------------------

# Initialising the emotion values
emotion_angry = 0
emotion_disgust = 0
emotion_fear = 0
emotion_happy = 0
emotion_sad = 0
emotion_surprise = 0
emotion_neutral = 0
total_frame = 0

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#---------------------------------------------------------------------------------------------------
### EMOTION INITIALIZATION PART END ------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------
### HEADPOSE INITIALIZATION PART START -------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

attention = 0
attention_perc = 0
#num_frames = 0



K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

#---------------------------------------------------------------------------------------------
### HEADPOSE INITIALIZATION PART END ---------------------------------------------------------
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
### HEADPOSE FUNCTIONS START -----------------------------------------------------------------
#---------------------------------------------------------------------------------------------

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def main():
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        num_frames+=1
        if ret:
            face_rects = detector(frame, 0)
#
            #if len(face_rects) > 0:
            for face_rect in face_rects:
                x,y,w,h = rect_to_bb(face_rect)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                shape = predictor(frame, face_rect)
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                #print("{:2.2f}".format(abs(euler_angle[1, 0])))
                if round(abs(euler_angle[0,0]))<=20.0 and round(abs(euler_angle[1,0]))<=20.0:
                    attention+=1

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



#---------------------------------------------------------------------------------------------
### HEADPOSE FUNCTIONS END -------------------------------------------------------------------
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
### MAIN CAMERA PROCESSING -------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

while(cap.isOpened()):
	ret, img = cap.read()
	if ret == True:
		#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')
		face_rects = detector(image, 0)
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

attention_perc=round((attention/total_frame)*100,2)
print("Attentive:"+str(attention_perc)+"%")


# Release the webcam
cap.release()
out.release()
cv2.destroyAllWindows()