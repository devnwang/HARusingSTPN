# Dense Optical FLow in OpenCV
# URL - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

import cv2 as cv
import numpy as np

path = "C:/Users/Nikki Wang/Pictures/error.gif"	# Insert video path here
cap = cv.VideoCapture(path)

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(1):
	ret, frame2 = cap.read()
	if ret:
		next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

		flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 6, 1.2, 0)

		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
		hsv[...,0] = ang*180/np.pi/2
		hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
		rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

		cv.imshow('frame2', rgb)
		k = cv.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			cv.imwrite('opticalfb.png', frame2)
			cv.imwrite('opticalhsv.png', rgb)
		prvs = next
	else:
		break

cap.release()
cv.destroyAllWindows()