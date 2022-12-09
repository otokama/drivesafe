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
import os

EYE_CLOSE_THRESHOLD = 0.5
YAWN_THRESHOLD = 0.9
EAR_THRESHOLD = 0.07
MAR_THRESHOLD = 0.7
NOD_CONSEQ_FRAMES = 5
NOD_THRESHOLD = 1

''' Calculates Eye Aspect Ratio. Formula taken from
Paper: Real-Time Eye Blink Detection using Facial Landmarks
by Tereza Soukupova and Jan Cech '''
def EAR(eye_coor):
	height_1 = np.linalg.norm(eye_coor[1] - eye_coor[5])
	height_2 = np.linalg.norm(eye_coor[2] - eye_coor[4])
	width = np.linalg.norm(eye_coor[0] - eye_coor[3])
	return (height_1 + height_2) / (width * 2)

def detectEyeClose(EAR, buffer):
	mean = np.mean(buffer)
	return mean - EAR >= EAR_THRESHOLD or EAR < 0.1

def detectYawn(MAR, buffer):
	mean = np.mean(buffer)
	return mean - MAR >= MAR_THRESHOLD or MAR > 0.8

# Calculates mouth aspect ratio. Similar to EAR formula.
def MOUTHAR(mouth_coor):
	height_1 = np.linalg.norm(mouth_coor[1] - mouth_coor[7])
	height_2 = np.linalg.norm(mouth_coor[2] - mouth_coor[6])
	height_3 = np.linalg.norm(mouth_coor[3] - mouth_coor[5])
	width = np.linalg.norm(mouth_coor[0] - mouth_coor[4])
	return (height_1 + height_2 + height_3) / (width * 3)

def calculateNoseY(nose):
	return np.sum(nose[:,1]) / nose.shape[0]

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=False, default='', help='path to input video')
	parser.add_argument('-p', '--face_landmark_predictor', required=True,
		help='path to shape_predictor_68_face_landmarks.dat')
	parser.add_argument('-v', '--verbose', action='store_true',
		help='turn on verbose mode: add EAR and nodding count, add face tracker')
	return parser.parse_args()

if __name__ == '__main__':
	input_args = parse_args()

	detector = dlib.get_frontal_face_detector()
	try:
		print('==> Loading face landmark predictor...')
		predictor = dlib.shape_predictor(input_args.face_landmark_predictor)
	except:
		sys.exit('==> Predictor model file not found! Exiting...\n\
	You can download the shape predictor model from here: \n\
	https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2')

	print('==> Finish loading...')
	(LEFT_IDX_START, LEFT_IDX_END) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
	(RIGHT_IDX_START, RIGHT_IDX_END) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
	(NOSE_IDX_START, NOSE_IDX_END) = face_utils.FACIAL_LANDMARKS_IDXS['nose']
	(MOUTH_IDX_START, MOUTH_IDX_END) = face_utils.FACIAL_LANDMARKS_IDXS['inner_mouth']

	videoInput = input_args.input != ''
	if videoInput:
		if not os.path.isfile(input_args.input):
			sys.exit('==> Video file not found. Exiting...')
		print('==> Playing input video file...')
		stream = FileVideoStream(path=input_args.input).start()
	else:
		print('==> Starting camera...')
		stream = VideoStream(src=0).start()

	frame = stream.read()

	buffer_len = 50
	EAR_buffer = [0.3] * buffer_len
	eyeIsClosed = False
	eyeClosedStartTime = 0
	eyesClosedInterval = 0
	maxEyesClosedInterval = 0
	yawnStartTime = 0
	yawnInterval = 0
	maxYawnInterval = 0
	prev_nose_y = 0
	nod_counter = 0
	total_nod = 0
	MAR_buffer = [0.01] * buffer_len
	# show_alert = False
	buffer_idx = 0


	
	pause = False
	cv2.namedWindow('Driver Drowsiness Detector')

	while True:
		if videoInput and not stream.more():
			break

		if pause:
			cv2.putText(frame, "Video paused. Press 'p' to continue.", (290 , 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 172, 49), 2)
			cv2.imshow("Driver Drowsiness Detector", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("p"):
				pause = False
			elif key == ord("q"):
				break
			continue

		frame = stream.read()
		try:
			frame = imutils.resize(frame, width=900)
		except AttributeError:
			break
		graysc_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(graysc_img, 0)

		if len(faces) == 1:
			face = faces[0]
			face_shape = predictor(graysc_img, face)
			face_shape = face_utils.shape_to_np(face_shape)

			leftEyeCoordinates = face_shape[LEFT_IDX_START:LEFT_IDX_END]
			rightEyeCoordinates = face_shape[RIGHT_IDX_START:RIGHT_IDX_END]
			nose = face_shape[NOSE_IDX_START:NOSE_IDX_END]
			mouth = face_shape[MOUTH_IDX_START:MOUTH_IDX_END]

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
			MAR = MOUTHAR(mouth)
			if buffer_idx < buffer_len:
				EAR_buffer[buffer_idx] = EAR_AVG
				MAR_buffer[buffer_idx] = MAR
				buffer_idx += 1
			else:
				eyeIsClosed = detectEyeClose(EAR_AVG, EAR_buffer)
				if eyeIsClosed and eyeClosedStartTime == 0:
					eyeClosedStartTime = time.time()
				elif not eyeIsClosed and eyeClosedStartTime > 0:
					eyesClosedInterval = time.time() - eyeClosedStartTime

					# print("==> Eyes closed interval: {:.3f}".format(eyesClosedInterval))
					eyeClosedStartTime = 0

				yawn = detectYawn(MAR, MAR_buffer)
				if yawn and yawnStartTime == 0:
					yawnStartTime = time.time()
				elif not yawn and yawnStartTime > 0:
					yawnInterval = time.time() - yawnStartTime
					# print("==> Yawn interval: {:.3f}".format(yawnInterval))
					yawnStartTime = 0

				maxEyesClosedInterval = max(maxEyesClosedInterval, eyesClosedInterval)
				maxYawnInterval = max(maxYawnInterval, yawnInterval)

			if input_args.verbose:
				face_bb = face_utils.rect_to_bb(face)
				face_rect = cv2.rectangle(frame, (face_bb[0], face_bb[1]), 
					(face_bb[0] + face_bb[2], face_bb[1] + face_bb[3]), (62, 255, 132), 2)
				cv2.putText(face_rect, 'Face', 
					(face_bb[0], face_bb[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (62, 255, 132), 2)
				cv2.putText(frame, "Nodding Cnt: {}".format(total_nod), (10, 30),
					cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
				cv2.putText(frame, "EAR (AVG): {:.2f}".format(EAR_AVG), (10, 60),
					cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
				mouthHull = cv2.convexHull(mouth)
				cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
				cv2.putText(frame, "MAR: {:.2f}".format(MAR), (10, 90),
					cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2 )

		elif len(faces) > 1:
			cv2.putText(frame, "Multiple faces detected. Only driver's face should be in the frame.",
				(100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (74, 95, 255), 2)
		else:
			cv2.putText(frame, "No face detected.",
				(360, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		if maxEyesClosedInterval > EYE_CLOSE_THRESHOLD or maxYawnInterval > YAWN_THRESHOLD:
			cv2.putText(frame, "Drowsiness Alert", (330, 440), cv2.FONT_HERSHEY_PLAIN, 1.7, (0, 0, 255), 2)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q") or key == ord("Q"):
			break
		elif key == ord("p") or key ==  ord("P"):
			pause = True
		cv2.imshow("Driver Drowsiness Detector", frame)

	print("==> Detector closing...")

	cv2.destroyAllWindows()
	stream.stop()