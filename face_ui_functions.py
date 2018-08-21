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
    cv2.line(frame, (int((nose[0][0]+left_eye[0][0])/2) ,left_eye[0][1]), (left_eye[1][0],left_eye[1][1]), (255, 255, 255),1)
    cv2.line(frame, (left_eye[1][0],left_eye[1][1]), (left_eye[3][0],left_eye[3][1]), (255, 255, 255),1)
    cv2.line(frame, (left_eye[3][0],left_eye[3][1]), (left_eye[5][0],left_eye[5][1]), (255, 255, 255),1)
    cv2.line(frame, (int((nose[0][0]+left_eye[0][0])/2) ,left_eye[0][1]), (left_eye[5][0],left_eye[5][1]), (255, 255, 255),1)
    
    cv2.line(frame, (right_eye[0][0],right_eye[0][1]), (right_eye[2][0],right_eye[2][1]), (255, 255, 255),1)
    cv2.line(frame, (right_eye[2][0],right_eye[2][1]), (int((nose[0][0]+right_eye[3][0])/2),right_eye[3][1]), (255, 255, 255),1)
    cv2.line(frame, (int((nose[0][0]+right_eye[3][0])/2),right_eye[3][1]), (right_eye[4][0],right_eye[4][1]), (255, 255, 255),1)
    cv2.line(frame, (right_eye[4][0],right_eye[4][1]), (right_eye[0][0],right_eye[0][1]), (255, 255, 255),1)
    cv2.line(frame, (mouth[0][0],mouth[0][1]), (right_eye[4][0],right_eye[4][1]), (255, 255, 255),1)
    cv2.line(frame, (mouth[6][0],mouth[6][1]), (left_eye[5][0],left_eye[5][1]), (255, 255, 255),1)
    cv2.line(frame, (mouth[0][0],mouth[0][1]), (nose[6][0],nose[6][1]), (255, 255, 255),1)
    cv2.line(frame, (mouth[6][0],mouth[6][1]), (nose[6][0],nose[6][1]), (255, 255, 255),1)
    cv2.line(frame, (mouth[0][0],mouth[0][1]), (mouth[9][0],mouth[9][1]), (255, 255, 255),1)
    cv2.line(frame, (mouth[6][0],mouth[6][1]), (mouth[9][0],mouth[9][1]), (255, 255, 255),1)

    cv2.line(frame, (nose[6][0],nose[6][1]), (nose[4][0],nose[4][1]), (255, 255, 255),1)
    cv2.line(frame, (nose[6][0],nose[6][1]), (nose[8][0],nose[8][1]), (255, 255, 255),1)

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

