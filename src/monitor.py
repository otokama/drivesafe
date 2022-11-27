from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import sys

threshold_change = False

# Calculates Eye Aspect Ratio. Formula taken from
# Paper: Real-Time Eye Blink Detection using Facial Landmarks
# by Tereza Soukupova and Jan Cech
def EAR(eye_coor) -> float:
	height_1 = np.linalg.norm(eye_coor[1] - eye_coor[5])
	height_2 = np.linalg.norm(eye_coor[2] - eye_coor[4])
	width = np.linalg.norm(eye_coor[0] - eye_coor[3])
	return (height_1 + height_2) / (width * 2)


BLINKS_CONSEQ_FRAMES = 3
BLINK_THRESHOLD = 0.17

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=False, default='', help='path to input video')
parser.add_argument('-p', '--face_landmark_predictor', required=True,
	help='path to shape_predictor_68_face_landmarks.dat')
input_args = parser.parse_args()



detector = dlib.get_frontal_face_detector()
try:
	predictor = dlib.shape_predictor(input_args.face_landmark_predictor)
except:
	sys.exit('Predictor model file not found! Exiting...\n\
You can download the shape predictor model from here: \n\
https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2')

(LEFT_IDX_START, LEFT_IDX_END) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(RIGHT_IDX_START, RIGHT_IDX_END) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

stream = VideoStream(src=0).start()

hasInput = input_args.input != ''
cv2.namedWindow('Drowsiness Detector')

blink_counter = 0
total_blinks = 0

while True:
	frame = stream.read()
	frame = imutils.resize(frame, width=500)

	graysc_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(graysc_img, 0)

	for face in faces:
		face_shape = predictor(graysc_img, face)
		face_shape = face_utils.shape_to_np(face_shape)
		leftEyeCoordinates = face_shape[LEFT_IDX_START:LEFT_IDX_END]
		rightEyeCoordinates = face_shape[RIGHT_IDX_START:RIGHT_IDX_END]
		EAR_AVG = (EAR(leftEyeCoordinates) + EAR(rightEyeCoordinates)) / 2.0

		if EAR_AVG < BLINK_THRESHOLD:
			blink_counter += 1
		else:
			if blink_counter >= BLINKS_CONSEQ_FRAMES:
				total_blinks += 1
			blink_counter = 0

		cv2.putText(frame, "Blinks: {}".format(total_blinks), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# cv2.putText(frame, "EAR: {:.2f}".format(EAR_AVG), (300, 30),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Drowsiness Detector", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break



cv2.destroyAllWindows()
stream.stop()