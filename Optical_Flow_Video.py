# Optical Flow Experiment
# Extracting a video's optical flow using OpenCV
# Code provided - URL: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

import torch
import torchvision
import numpy as np
import cv2 as cv  

def opt_flow():
	path = "/kick.avi" # Insert video path here
	cap = cv.VideoCapture(path)

	# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 100,
						   qualityLevel = 0.3,
						   minDistance = 7,
						   blockSize = 7 )

	# Parameters for lucas kanade optical flow
	lk_params = dict ( winSize = (15, 15),
					   maxLevel = 2,
					   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

	# Create some random colors
	#color = np.random.randint(0, 255, (100, 3))
	color = (0, 255, 0)

	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
	# Used to decide points to track
	p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	# Create a mask image for drawing purposes
	#mask = np.zeros_like(old_frame)

	optFrames = []
	rgbFrames = []
	frameNum = 0
	while(1):
		ret, frame = cap.read()
		if ret:
			frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

			# Resize frame to be 3 x 299 x 299

			# calculate optical flow
			# Pass in previous frame, previous poitns, and next frame
			# Returns next points + status numbers: 1 - found, 0 - not found
			p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

			# Select good points
			good_new = p1[st==1]
			good_old = p0[st==1]

			# Create a mask image for drawing purposes
			mask = np.zeros_like(old_frame)

			# draw the tracks
			for i, (new, old) in enumerate(zip(good_new, good_old)):
				a,b = new.ravel()
				c,d = old.ravel()
				#mask = cv.arrowedLine(mask, (a, b), (c, d), color[i].tolist(), 2)
				mask = cv.arrowedLine(mask, (a, b), (c, d), color, 2)
				#cv.imshow("mask", mask)
				#frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
				frame = cv.circle(frame, (a, b), 5, color, -1)

			img = cv.add(frame, mask)
			# maskname = "mask{}.png".format(frameNum)
			#filename = "frame{}.png".format(frameNum)
			tensor = torch.from_numpy(img)
			rgbTensor = torch.from_numpy(frame)
			#arrFrames[frameNum] = tensor
			optFrames.append(tensor)
			rgbFrames.append(rgbTensor)
			frameNum += 1
			# if x == 2:
			# 	cv.imwrite(filename, img)
			# cv.imwrite(maskname, mask)

			# cv.imshow('frame', img)
			k = cv.waitKey(30) & 0xff
			if k == 27:
				break
		else:
			break

		# Now update the previous frame and previous points
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1, 1, 2)
		# print(arrFrames)
		
	cv.destroyAllWindows()
	cap.release()
	return arrFrames

if __name__ == '__main__':
    arrFrames = opt_flow()