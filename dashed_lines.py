import cv2
 
img = cv2.imread('lena-eyes.png')
 
# (1) create a copy of the original:
overlay = img.copy()
# (2) draw shapes:
cv2.circle(overlay, (133, 132), 12, (255, 255, 0), -1)
cv2.circle(overlay, (166, 132), 12, (127, 255, 0), -1)
cv2.circle(overlay, (216, 182), 12, (127, 255, 0), -1)

cv2.rectangle(overlay, (166, 132),(216, 182), (127, 255, 0), -1)
# (3) blend with the original:
opacity = 0.4
cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
 
# display result (press 'q' to quit):
cv2.namedWindow('Transparency')
cv2.imshow('Transparency', img)
while (cv2.waitKey() & 0xff) != ord('q'): 
    pass

cv2.destroyAllWindows()
