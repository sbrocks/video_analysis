import cv2
import sys
import dlib
import numpy as np
import argparse
from imutils import face_utils
import imutils
from scipy.spatial.distance import euclidean as dist

# Run this file as such 'python opencv_blink_detect.py -p sp.dat'

#Defining EAR
##EAR is the ratio between width and height of eye
EYE_AR_THRESH = 0.43
EYE_AR_CONSEC_FRAMES = 1

L_COUNTER = 0  # Counts number of frame left eye has been closed
R_COUNTER = 0  # Counts number of frame right eye has been closed

TOTAL_BLINK_COUNTER = 0
L_BLINK_COUNTER = 0
R_BLINK_COUNTER = 0

#Method to convert rectanngle to Bounding Box
def rect2bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

#Converting Shape to numpy array
def shape2np(shape, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates


def draw_details(rect, shape, image):
    #Rectangle over the face
    (x, y, w, h) = rect2bb(rect)

    #Points of the facial landmarks
    shape = shape2np(shape)

    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

    #Drawing rectangle over the face
    #Drawing a point over each landmark
    #for (x,y) in shape:
        #cv2.circle(image, (x,y), 2, (0, 0, 255), -1)
    return image


def find_features(image):
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shapes = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shapes.append(shape)
        image = draw_details(rect, shape, image)
    return image, rects, shapes


def calculate_EAR(eye):
    if len(eye)==6:
        width = dist(eye[0],eye[3])
        A = dist(eye[1],eye[5])
        B = dist(eye[2],eye[4])
        EAR = (A+B)/width
        return EAR
    else:
        print("Error in eye shape")
        return -1

def calculate_frown(l_eyebrow,r_eyebrow):
    if len(l_eyebrow)==5 and len(r_eyebrow)==5:
        frown_width = dist(l_eyebrow[0],r_eyebrow[4])
        return frown_width
    else:
        print("Error in eyebrow shape")
        return -1

def calculate_mouth(mouth):
    if len(mouth)==20:
        mouth_dist = dist(mouth[14],mouth[18])
        return mouth_dist
    else:
        print("Error in mouth shape")
        return -1

def calculate_enthusiasm(l_eye,r_eye,l_eyebrow,r_eyebrow):
    if len(l_eye)==6 and len(r_eye)==6 and len(l_eyebrow)==5 and len(r_eyebrow)==5:
        r_enthu_dist = dist(r_eye[2],r_eyebrow[3])
        l_enthu_dist = dist(l_eye[2],l_eyebrow[3])
        enthusiasm_dist = (r_enthu_dist+l_enthu_dist)/2.0
        return enthusiasm_dist
    else:
        print("Error in eye or eyebrow shape")
        return -1




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



def get_eyes(features):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    features = shape2np(features)
    l_eye = features[lStart:lEnd]
    r_eye = features[rStart:rEnd]
    return l_eye, r_eye

def get_eyebrows(features):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    features = shape2np(features)
    l_eyebrow = features[lStart:lEnd]
    r_eyebrow = features[rStart:rEnd]
    #print(l_eyebrow)
    #print(r_eyebrow)
    return l_eyebrow, r_eyebrow

def get_mouth(features):
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    features = shape2np(features)
    mouth = features[mStart:mEnd]
    #print(mouth)
    return mouth

def convert_arc(pt1, pt2, sagitta):

    # extract point coordinates
    x1, y1 = pt1
    x2, y2 = pt2

    # find normal from midpoint, follow by length sagitta
    n = np.array([y2 - y1, x1 - x2])
    n_dist = np.sqrt(np.sum(n**2))

    if np.isclose(n_dist, 0):
        # catch error here, d(pt1, pt2) ~ 0
        print('Error: The distance between pt1 and pt2 is too small.')

    n = n/n_dist
    x3, y3 = (np.array(pt1) + np.array(pt2))/2 + sagitta * n

    # calculate the circle from three points
    # see https://math.stackexchange.com/a/1460096/246399
    A = np.array([
        [x1**2 + y1**2, x1, y1, 1],
        [x2**2 + y2**2, x2, y2, 1],
        [x3**2 + y3**2, x3, y3, 1]])
    M11 = np.linalg.det(A[:, (1, 2, 3)])
    M12 = np.linalg.det(A[:, (0, 2, 3)])
    M13 = np.linalg.det(A[:, (0, 1, 3)])
    M14 = np.linalg.det(A[:, (0, 1, 2)])

    if np.isclose(M11, 0):
        # catch error here, the points are collinear (sagitta ~ 0)
        print('Error: The third point is collinear.')

    cx = 0.5 * M12/M11
    cy = -0.5 * M13/M11
    radius = np.sqrt(cx**2 + cy**2 + M14/M11)

    # calculate angles of pt1 and pt2 from center of circle
    pt1_angle = 180*np.arctan2(y1 - cy, x1 - cx)/np.pi
    pt2_angle = 180*np.arctan2(y2 - cy, x2 - cx)/np.pi

    return (cx, cy), radius, pt1_angle, pt2_angle

def draw_ellipse(img, center, axes, angle,startAngle, endAngle, color,thickness=1, lineType=cv2.LINE_AA, shift=10):
    # uses the shift to accurately get sub-pixel resolution for arc
    # taken from https://stackoverflow.com/a/44892317/5087436
    center = (int(round(center[0] * 2**shift)),int(round(center[1] * 2**shift)))
    axes = (int(round(axes[0] * 2**shift)), int(round(axes[1] * 2**shift)))
    return cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift)



def face_ui(frame,features):
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (rebStart, rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (lebStart, lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    features = shape2np(features)
    mouth = features[mStart:mEnd]
    right_eyebrow = features[rebStart:rebEnd]
    left_eyebrow = features[lebStart:lebEnd]
    right_eye = features[reStart:reEnd]
    left_eye = features[leStart:leEnd]
    nose = features[nStart:nEnd]
    jaw = features[jStart:jEnd]


    # Lines start    
    cv2.line(frame, (jaw[8][0],jaw[8][1]), (jaw[8][0],nose[0][1]-50), (255, 255, 255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[8][0]-3,nose[0][1]-40),(jaw[8][0]+3,nose[0][1]-40),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.rectangle(frame,(jaw[8][0]-5,jaw[8][1]-5),(jaw[8][0]+5,jaw[8][1]+5),(255,255,255),2,lineType=cv2.LINE_AA)
    
    cv2.rectangle(frame,(jaw[8][0]-5,nose[0][1]-5),(jaw[8][0]+5,nose[0][1]+5),(255,255,255),2,lineType=cv2.LINE_AA)
    
    cv2.rectangle(frame,(jaw[8][0]-3,nose[6][1]-3),(jaw[8][0]+3,nose[6][1]+3),(255,255,255),1,lineType=cv2.LINE_AA)

    # Cross sign
    # Jaw parts
    cv2.line(frame,(jaw[10][0],jaw[10][1]-3),(jaw[10][0],jaw[10][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[10][0]-3,jaw[10][1]),(jaw[10][0]+3,jaw[10][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[11][0]-3,jaw[11][1]),(jaw[11][0]+3,jaw[11][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[11][0],jaw[11][1]-3),(jaw[11][0],jaw[11][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[13][0]-3,jaw[13][1]),(jaw[13][0]+3,jaw[13][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[13][0],jaw[13][1]-3),(jaw[13][0],jaw[13][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[6][0]-3,jaw[6][1]),(jaw[6][0]+3,jaw[6][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[6][0],jaw[6][1]-3),(jaw[6][0],jaw[6][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[5][0]-3,jaw[5][1]),(jaw[5][0]+3,jaw[5][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[5][0],jaw[5][1]-3),(jaw[5][0],jaw[5][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[3][0],jaw[3][1]-3),(jaw[3][0],jaw[3][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[3][0]-3,jaw[3][1]),(jaw[3][0]+3,jaw[3][1]),(255,255,255),2,lineType=cv2.LINE_AA)

    cv2.rectangle(frame,(jaw[2][0]-5,jaw[2][1]-5),(jaw[2][0]+5,jaw[2][1]+5),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.rectangle(frame,(jaw[14][0]-5,jaw[14][1]-5),(jaw[14][0]+5,jaw[14][1]+5),(255,255,255),2,lineType=cv2.LINE_AA)

    # Mouth part
    cv2.line(frame,(mouth[13][0]-3,mouth[13][1]),(mouth[13][0]+3,mouth[13][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(mouth[13][0],mouth[13][1]-3),(mouth[13][0],mouth[13][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(mouth[15][0]-3,mouth[15][1]),(mouth[15][0]+3,mouth[15][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(mouth[15][0],mouth[15][1]-3),(mouth[15][0],mouth[15][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)

    cv2.rectangle(frame,(mouth[6][0]-5,mouth[6][1]-5),(mouth[6][0]+5,mouth[6][1]+5),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.rectangle(frame,(mouth[0][0]-5,mouth[0][1]-5),(mouth[0][0]+5,mouth[0][1]+5),(255,255,255),1,lineType=cv2.LINE_AA)

    # Extra stars
    cv2.line(frame,(mouth[13][0]-3,mouth[13][1]),(mouth[13][0]+3,mouth[13][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(mouth[13][0],mouth[13][1]-3),(mouth[13][0],mouth[13][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)

    cv2.line(frame,(nose[8][0]-3,nose[8][1]),(nose[8][0]+3,nose[8][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(nose[8][0],nose[8][1]-3),(nose[8][0],nose[8][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)


    # Extra Lines 
    cv2.line(frame,(left_eye[4][0]-3,left_eye[4][1]+20),(left_eye[4][0]+3,left_eye[4][1]+20),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eye[4][0],left_eye[4][1]+17),(left_eye[4][0],left_eye[4][1]+23),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eye[3][0],left_eye[3][1]+17),(left_eye[3][0],left_eye[3][1]+23),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eye[3][0]-3,left_eye[3][1]+20),(left_eye[3][0]+3,left_eye[3][1]+20),(255,255,255),2,lineType=cv2.LINE_AA)

    
    cv2.line(frame,(jaw[8][0]+3,mouth[9][1]),(jaw[8][0]-3,mouth[9][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[6][0],jaw[6][1]),(jaw[7][0],mouth[9][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[7][0],mouth[9][1]),(jaw[8][0],mouth[9][1]+5),(255,255,255),1,lineType=cv2.LINE_AA)
    
    cv2.line(frame,(jaw[7][0],mouth[9][1]),(mouth[0][0],mouth[0][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[8][0],mouth[9][1]+10),(jaw[7][0]-6,mouth[9][1]+12),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[8][0],jaw[8][1]),(jaw[8][0]-9,jaw[7][1]-17),(255,255,255),1,lineType=cv2.LINE_AA)
    
    cv2.line(frame,(jaw[7][0]-6,mouth[9][1]+12),(jaw[8][0]-9,jaw[7][1]-17),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[6][0],jaw[6][1]),(jaw[8][0]-9,jaw[7][1]-17),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[8][0]-11,jaw[8][1]+1),(jaw[8][0]-9,jaw[7][1]-17),(255,255,255),1,lineType=cv2.LINE_AA)
    
    cv2.line(frame,(jaw[8][0]-22,jaw[8][1]),(jaw[8][0]-9,jaw[7][1]-17),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[8][0]-22,jaw[8][1]),(jaw[8][0]-9,jaw[7][1]-17),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[8][0],jaw[8][1]-20),(jaw[8][0]-9,jaw[7][1]-17),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[8][0],mouth[9][1]+10),(jaw[8][0]-9,jaw[7][1]-17),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[7][0],mouth[9][1]),(mouth[0][0],mouth[0][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[7][0],mouth[9][1]),(jaw[6][0]-7,mouth[9][1]+4),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(mouth[0][0],mouth[0][1]),(jaw[6][0]-7,mouth[9][1]+4),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[7][0]-6,mouth[9][1]+12),(jaw[6][0]-7,mouth[9][1]+4),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[5][0],mouth[0][1]+4),(jaw[6][0]-7,mouth[9][1]+4),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[5][0],mouth[0][1]+4),(mouth[0][0],mouth[0][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    # Above mouth line design
    cv2.line(frame,(jaw[5][0],mouth[0][1]+4),(jaw[5][0]-6,mouth[2][1]-2),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(mouth[0][0],mouth[0][1]),(mouth[0][0]-2,mouth[2][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[5][0]-6,mouth[2][1]-2),(mouth[0][0]-2,mouth[2][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(mouth[0][0]-2,mouth[2][1]),(jaw[5][0],mouth[0][1]+4),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[5][0]-6,mouth[2][1]-2),(mouth[0][0],mouth[0][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[5][0]-6,mouth[2][1]-2),(jaw[4][0],nose[3][1]+2),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(mouth[0][0]-2,mouth[2][1]),(mouth[0][0]-6,nose[3][1]+6),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(mouth[0][0]-6,nose[3][1]+6),(jaw[4][0]+6,nose[3][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[4][0],nose[3][1]+2),(jaw[4][0]+6,nose[3][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[4][0],nose[3][1]+2),(jaw[5][0]+1,nose[3][1]+9),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[5][0]+1,nose[3][1]+9),(mouth[0][0]-6,nose[3][1]+6),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[5][0]+1,nose[3][1]+9),(jaw[5][0]-6,mouth[2][1]-2),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[5][0]+1,nose[3][1]+9),(mouth[0][0]-2,mouth[2][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(mouth[0][0],mouth[0][1]),(nose[4][0]-5,mouth[2][1]-7),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(mouth[0][0]-2,mouth[2][1]),(nose[4][0]-5,mouth[2][1]-7),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(nose[4][0]-5,mouth[2][1]-7),(nose[4][0]-8,mouth[2][1]-10),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(nose[4][0]-8,mouth[2][1]-10),(mouth[0][0]-2,mouth[2][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(nose[4][0]-8,mouth[2][1]-10),(mouth[0][0]-6,nose[3][1]+6),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(nose[4][0]-5,mouth[2][1]-10),(nose[4][0]-11,mouth[2][1]-10),(255,255,255),2)
    cv2.line(frame,(nose[4][0]-8,mouth[2][1]-7),(nose[4][0]-8,mouth[2][1]-13),(255,255,255),2)

    # Above nose design part
    cv2.line(frame,(jaw[4][0],nose[3][1]+2),(jaw[4][0]+2,nose[3][1]-5),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[4][0]+6,nose[3][1]),(jaw[4][0]+2,nose[3][1]-5),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[4][0]+2,nose[3][1]-5),(jaw[4][0]+13,nose[2][1]+1),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[4][0]+6,nose[3][1]),(jaw[4][0]+13,nose[2][1]+1),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(mouth[0][0]-6,nose[3][1]+6),(jaw[4][0]+13,nose[2][1]+1),(255,255,255),1,lineType=cv2.LINE_AA)

    # Eye design part
    cv2.line(frame,(jaw[4][0]+2,nose[3][1]-5),(right_eye[0][0]-13,right_eye[0][1]-1),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eye[0][0]-13,right_eye[0][1]-1),(right_eye[0][0],right_eye[0][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(jaw[4][0]+13,nose[2][1]+1),(right_eye[0][0],right_eye[0][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    # Nose part
    cv2.line(frame,(nose[4][0],nose[4][1]-15),(mouth[0][0]-6,nose[3][1]+6),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(nose[4][0],nose[4][1]-15),(nose[4][0]-8,mouth[2][1]-10),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(nose[4][0],nose[4][1]-15),(right_eye[3][0]+3,right_eye[3][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eye[3][0]+3,right_eye[3][1]),(mouth[0][0]-6,nose[3][1]+6),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(jaw[4][0]+13,nose[2][1]+1),(right_eye[2][0],nose[2][1]+1),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eye[3][0]+3,right_eye[3][1]),(right_eye[2][0],nose[2][1]+1),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eye[2][0],nose[2][1]+1),(mouth[0][0]-6,nose[3][1]+6),(255,255,255),1,lineType=cv2.LINE_AA)



    # Right eye designs
    cv2.line(frame,(right_eyebrow[2][0],right_eyebrow[2][1]),(jaw[8][0],right_eyebrow[2][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[2][0],right_eyebrow[2][1]),(jaw[8][0],nose[0][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[4][0]-1,nose[0][1]),(jaw[8][0],nose[0][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[4][0]-1,nose[0][1]),(right_eyebrow[4][0]-1,nose[0][1]-9),(255,255,255),1,lineType=cv2.LINE_AA)


    # Right eyebrows
    cv2.line(frame,(right_eyebrow[0][0],right_eyebrow[0][1]-3),(right_eyebrow[0][0],right_eyebrow[0][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[0][0]-3,right_eyebrow[0][1]),(right_eyebrow[0][0]+3,right_eyebrow[0][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[1][0],right_eyebrow[1][1]-3),(right_eyebrow[1][0],right_eyebrow[1][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[1][0]-3,right_eyebrow[1][1]),(right_eyebrow[1][0]+3,right_eyebrow[1][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[2][0],right_eyebrow[2][1]-3),(right_eyebrow[2][0],right_eyebrow[2][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[2][0]-3,right_eyebrow[2][1]),(right_eyebrow[2][0]+3,right_eyebrow[2][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[3][0],right_eyebrow[3][1]+10),(right_eyebrow[3][0],right_eyebrow[3][1]+4),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[3][0]-3,right_eyebrow[3][1]+7),(right_eyebrow[3][0]+3,right_eyebrow[3][1]+7),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[4][0]-1,nose[0][1]-3),(right_eyebrow[4][0]-1,nose[0][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eyebrow[4][0]-4,nose[0][1]),(right_eyebrow[4][0]+2,nose[0][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    
    cv2.line(frame,(right_eye[3][0],right_eye[3][1]),(left_eye[0][0],right_eye[3][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(right_eye[3][0],right_eye[3][1]-3),(right_eye[3][0],right_eye[3][1]+3),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eye[3][0]-3,right_eye[3][1]),(right_eye[3][0]+3,right_eye[3][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eye[0][0],right_eye[3][1]-3),(left_eye[0][0],right_eye[3][1]+3),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eye[0][0]-3,right_eye[3][1]),(left_eye[0][0]+3,right_eye[3][1]),(255,255,255),1,lineType=cv2.LINE_AA)

    cv2.line(frame,(right_eye[0][0]-3,right_eye[0][1]),(right_eye[0][0]+3,right_eye[0][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(right_eye[0][0],right_eye[0][1]-3),(right_eye[0][0],right_eye[0][1]+3),(255,255,255),1,lineType=cv2.LINE_AA)



    # Left eyebrows
    cv2.line(frame,(left_eyebrow[0][0],left_eyebrow[0][1]-3),(left_eyebrow[0][0],left_eyebrow[0][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eyebrow[0][0]-3,left_eyebrow[0][1]),(left_eyebrow[0][0]+3,left_eyebrow[0][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eyebrow[1][0],left_eyebrow[1][1]-3),(left_eyebrow[1][0],left_eyebrow[1][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eyebrow[1][0]-3,left_eyebrow[1][1]),(left_eyebrow[1][0]+3,left_eyebrow[1][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eyebrow[2][0],left_eyebrow[2][1]-3),(left_eyebrow[2][0],left_eyebrow[2][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eyebrow[2][0]-3,left_eyebrow[2][1]),(left_eyebrow[2][0]+3,left_eyebrow[2][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eyebrow[3][0],left_eyebrow[3][1]-3),(left_eyebrow[3][0],left_eyebrow[3][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eyebrow[3][0]-3,left_eyebrow[3][1]),(left_eyebrow[3][0]+3,left_eyebrow[3][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eyebrow[4][0],left_eyebrow[4][1]-3),(left_eyebrow[4][0],left_eyebrow[4][1]+3),(255,255,255),2,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eyebrow[4][0]-3,left_eyebrow[4][1]),(left_eyebrow[4][0]+3,left_eyebrow[4][1]),(255,255,255),2,lineType=cv2.LINE_AA)
    
    cv2.line(frame,(left_eye[2][0]-3,left_eye[2][1]),(left_eye[2][0]+3,left_eye[2][1]),(255,255,255),1,lineType=cv2.LINE_AA)
    cv2.line(frame,(left_eye[2][0],left_eye[2][1]-3),(left_eye[2][0],left_eye[2][1]+3),(255,255,255),1,lineType=cv2.LINE_AA)

    # Curves
    pt1 = (jaw[8][0],jaw[8][1])
    pt2 = (jaw[6][0],jaw[6][1])
    sagitta =5

    center, radius, start_angle, end_angle = convert_arc(pt1, pt2, sagitta)
    axes = (radius, radius)
    draw_ellipse(frame, center, axes, 0, start_angle, end_angle, (255,255,255))

    #cv2.ellipse(frame,center,axes,0,start_angle,end_angle,255,2)
    #cv2.line(frame, (jaw[8][0],jaw[8][1]), (jaw[8][0],nose[0][1]-30), (255, 255, 255),2)

    #cv2.line(frame, (int((nose[0][0]+left_eye[0][0])/2) ,left_eye[0][1]), (left_eye[1][0],left_eye[1][1]), (255, 255, 255),1)
    #cv2.line(frame, (left_eye[1][0],left_eye[1][1]), (left_eye[3][0],left_eye[3][1]), (255, 255, 255),1)

    ###################################################################
    # Right
    ###################################################################

    # Adding transparent shapes
    overlay = frame.copy()
    #cv2.circle(overlay, (int((right_eye[0][0]+right_eye[3][0])/2) , int((right_eye[0][1]+right_eye[3][1])/2) ), 12, (255, 255, 0), -1)

    # Emotion Recogniton
    cv2.rectangle(overlay, (jaw[0][0]-50 , jaw[0][1]-50 ),( jaw[0][0]-130 , jaw[0][1]-100 ), (127, 255, 0), -1)
    cv2.putText(overlay,"Emotion",( jaw[0][0]-115 , jaw[0][1]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,147,255),1,cv2.LINE_AA)
    cv2.putText(overlay,"Recognition",( jaw[0][0]-125 , jaw[0][1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,147,255),1,cv2.LINE_AA)
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)


    # Speech
    cv2.rectangle(overlay, (jaw[0][0]-50 , jaw[0][1]-10 ),( jaw[0][0]-130 , jaw[0][1]+40 ), (127, 255, 0), -1)
    cv2.putText(overlay,"Speech",( jaw[0][0]-115 , jaw[0][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,147,255),1,cv2.LINE_AA)
    #cv2.putText(overlay,"Recognition",( jaw[0][0]-125 , jaw[0][1]-65), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,147,255),1,cv2.LINE_AA)
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # Facial Analysis
    cv2.rectangle(overlay, (jaw[0][0]-50 , jaw[0][1]+80 ),( jaw[0][0]-130 , jaw[0][1]+130 ), (127, 255, 0), -1)
    cv2.putText(overlay,"Facial",( jaw[0][0]-110 , jaw[0][1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,147,255),1,cv2.LINE_AA)
    cv2.putText(overlay,"Analysis",( jaw[0][0]-115 , jaw[0][1]+120), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,147,255),1,cv2.LINE_AA)
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)


    ###################################################################
    # Left
    ###################################################################
    
    # Smile design
    #cv2.line(frame,(mouth[6][0],mouth[6][1]), (mouth[6][0]+50,mouth[6][1]),(127,0,255),1)

    cv2.rectangle(overlay, (left_eye[3][0] +80, mouth[6][1]-15 ),( left_eye[3][0]+180 , mouth[6][1]+15 ), (127, 255, 0), -1)
    cv2.putText(overlay,'Smile',( left_eye[3][0]+115 , mouth[6][1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,47,255),1,cv2.LINE_AA)
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # Blink
    #cv2.line(frame,(left_eye[3][0],left_eye[3][1]), (left_eye[3][0]+50,left_eye[3][1]),(127,0,255),1)

    cv2.rectangle(overlay, (left_eye[3][0] +80, left_eye[3][1]-15 ),( left_eye[3][0]+180 , left_eye[3][1]+15 ), (127, 255, 0), -1)
    cv2.putText(overlay,'Blink',( left_eye[3][0]+115 , left_eye[3][1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,47,255),1,cv2.LINE_AA)
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # Communication
    #cv2.line(frame,(left_eye[3][0],left_eye[3][1]), (left_eye[3][0]+50,left_eye[3][1]),(127,0,255),1)

    cv2.rectangle(overlay, (left_eye[3][0] +80, left_eye[3][1]-60 ),( left_eye[3][0]+180 , left_eye[3][1]-30 ), (127, 255, 0), -1)
    cv2.putText(overlay,'Positive',( left_eye[3][0]+105 , left_eye[3][1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,47,255),1,cv2.LINE_AA)
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # Positive Attitude
    #cv2.line(frame,(left_eye[3][0],left_eye[3][1]), (left_eye[3][0]+50,left_eye[3][1]),(127,0,255),1)

    cv2.rectangle(overlay, (left_eye[3][0] +80, mouth[6][1]+60 ),( left_eye[3][0]+180 , mouth[6][1]+30 ), (127, 255, 0), -1)
    cv2.putText(overlay,'Communication',( left_eye[3][0]+80 , mouth[6][1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,47,255),1,cv2.LINE_AA)
    opacity = 0.4
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)



face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

#------------------------------------------------------------------------------------------------------
### HEAD POSE INITIALIZATION PART START -------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

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



cam = cv2.VideoCapture("Rose2.mp4")

""" We use cv2.VideoWriter() to save the video """
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('emotion_shrofile.avi',fourcc,20.0,(640,480))


attention = 0
attention_perc = 0
num_frames=0
frown_count=0
enthusiasm_count=0
frame_no=0
x=list()
y=list()
TOTAL_TIME_VIDEO = 0

while cam.isOpened():
    ret, img = cam.read()
    if ret==True:
        TOTAL_TIME_VIDEO = cam.get(cv2.CAP_PROP_POS_MSEC)
        num_frames+=1
        img, rects, feature_array = find_features(img)
        #n_faces = len(rects)
        #print(n_faces)
        for (i, rect) in enumerate(rects):
            features = feature_array[i]         # Currently only calculating blink for the First face
            l_eye, r_eye = get_eyes(features)
            l_eyebrow,r_eyebrow = get_eyebrows(features)
            mouth = get_mouth(features)
            
            # Head Pose Code
            tx,ty,tw,th = rect2bb(rect)
            #cv2.rectangle(img,(tx,ty),(tx+tw,ty+th),(255,0,0),2)
            h_shape = predictor(img,rect)
            h_shape = face_utils.shape_to_np(h_shape)
            reprojectdst, euler_angle = get_head_pose(h_shape)



            # Make UI for Face
            face_ui(img,features)

            l_EAR = calculate_EAR(l_eye)
            r_EAR = calculate_EAR(r_eye)

            frown_dist = calculate_frown(l_eyebrow,r_eyebrow)
            mouth_dist = calculate_mouth(mouth)
            enthusiasm_dist = calculate_enthusiasm(l_eye,r_eye,l_eyebrow,r_eyebrow)

            if round(abs(euler_angle[0,0]))<=20.0 and round(abs(euler_angle[1,0]))<=20.0:
                attention+=1

            #print("Enthusiasm:"+str(enthusiasm_dist))
            if mouth_dist<12.0 and enthusiasm_dist>20.0:
                enthusiasm_count+=1

            #print("Frown Dist:"+str(frown_dist))
            if frown_dist < 16.0:
                frown_count+=1

            L_COUNTER += l_EAR <= EYE_AR_THRESH
            R_COUNTER += r_EAR <= EYE_AR_THRESH

            eye_aspect_ratio = (l_EAR+r_EAR)/2.0
            #print(eye_aspect_ratio)
            if L_COUNTER == EYE_AR_CONSEC_FRAMES:
                L_COUNTER = 0
                TOTAL_BLINK_COUNTER += 1  # Blink has been Detected in the Left eye
                L_BLINK_COUNTER += 1
            if R_COUNTER == EYE_AR_CONSEC_FRAMES:
                R_COUNTER = 0
                TOTAL_BLINK_COUNTER += 1  # Blink has been Detected in the  Right eye
                R_BLINK_COUNTER += 1

            
            cv2.putText(img, "Blinks: {} , {} ".format(L_BLINK_COUNTER, R_BLINK_COUNTER), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "EAR: {:.2f} , {:.2f} ".format(l_EAR,r_EAR), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            x.append(frame_no);
            y.append(l_EAR);
        #out.write(img)
        cv2.imshow('my webcam', img)
        waitKey = cv2.waitKey(1)
        if waitKey == 27: #Escape clicked.Exit program
            break
        elif waitKey == 114:#'R' Clicked.Reset Counter 
            L_BLINK_COUNTER = 0
            R_BLINK_COUNTER = 0
    else:
        break


# Frown Calculations
print("Frown Percentage: "+str(round((frown_count/num_frames*100),2))+"%")
print("Enthusiasm Percentage: "+str(round((enthusiasm_count/num_frames*100),2))+"%")
attention_perc=round((attention/num_frames)*100,2)
print("Attentive:"+str(attention_perc)+"%")
# Blink Calculations
TOTAL_TIME_VIDEO_SECS = int(TOTAL_TIME_VIDEO/1000)
TOTAL_BLINK_COUNTER = int(TOTAL_BLINK_COUNTER/2)
#print(TOTAL_BLINK_COUNTER)
#print(TOTAL_TIME_VIDEO_SECS)

TOTAL_TIME_VIDEO_MINS = TOTAL_TIME_VIDEO_SECS/60
TOTAL_TIME_VIDEO_EXTRA_SECS = TOTAL_TIME_VIDEO_SECS%60

# Ideal Blinking Rate is 15-20 blinks (per minute)
# 15 blinks : Normal , Nervous : Very low
# 20 blinks :        , Nervous : 25-40%
# 25 blinks :        , Nervous : >50% -75%
# 30 blinks :        , Nervous : 75-100%

#if TOTAL_BLINK_COUNTER > TOTAL_TIME_VIDEO_SECS/4

# Release the webcam
cam.release()
#out.release()
cv2.destroyAllWindows()

