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

def calculateNoseY(nose) -> float:
	return np.sum(nose[:,1]) / nose.shape[0]

NOD_CONSEQ_FRAMES = 4
NOD_THRESHOLD = 1
BLINKS_CONSEQ_FRAMES = 4
BLINK_THRESHOLD = 0.17
ALERT_FRAME_DURATION = 80

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
(NOSE_IDX_START, NOSE_IDX_END) = face_utils.FACIAL_LANDMARKS_IDXS['nose']

videoInput = input_args.input != ''
if videoInput:
	stream = FileVideoStream(path=input_args.input).start()
else:
	stream = VideoStream(src=0).start()
cv2.namedWindow('Drowsiness Detector')

blink_counter = 0
total_blinks = 0
prev_nose_y = 0
nod_counter = 0
total_nod = 0
alertFrameCounts = ALERT_FRAME_DURATION
show_alert = False


while True:
	if videoInput and not stream.more():
		break
	frame = stream.read()
	frame = imutils.resize(frame, width=500)

	graysc_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(graysc_img, 0)

	for i, face in enumerate(faces):
		# face_bb = face_utils.rect_to_bb(face)

		# face_rect = cv2.rectangle(frame, (face_bb[0], face_bb[1]), 
		# 	(face_bb[0] + face_bb[2], face_bb[1] + face_bb[3]), (62, 255, 132), 2)
		# cv2.putText(face_rect, 'Face' + str(i + 1), 
		# 	(face_bb[0], face_bb[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (62, 255, 132))


		face_shape = predictor(graysc_img, face)
		face_shape = face_utils.shape_to_np(face_shape)

		leftEyeCoordinates = face_shape[LEFT_IDX_START:LEFT_IDX_END]
		rightEyeCoordinates = face_shape[RIGHT_IDX_START:RIGHT_IDX_END]
		nose = face_shape[NOSE_IDX_START:NOSE_IDX_END]

		curr_nose_y = calculateNoseY(nose)
		if prev_nose_y != 0:
			if curr_nose_y - prev_nose_y > NOD_THRESHOLD:
				nod_counter += 1
			else:
				if nod_counter > NOD_CONSEQ_FRAMES:
					total_nod += 1
				nod_counter = 0


		prev_nose_y = curr_nose_y

		EAR_AVG = (EAR(leftEyeCoordinates) + EAR(rightEyeCoordinates)) / 2.0

		if EAR_AVG < BLINK_THRESHOLD:
			blink_counter += 1
		else:
			if blink_counter >= BLINKS_CONSEQ_FRAMES:
				total_blinks += 1
			blink_counter = 0


		cv2.putText(frame, "Blinks: {}".format(total_blinks), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.putText(frame, "Nodding Cnt: {}".format(total_nod), (10, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.putText(frame, "EAR (AVG): {:.2f}".format(EAR_AVG), (10, 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



	cv2.imshow("Drowsiness Detector", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break



cv2.destroyAllWindows()
stream.stop()