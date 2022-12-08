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

EYE_CLOSE_THRESHOLD = 0.09
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

def eyeClosed(EAR, buffer):
	EAR_mean = np.mean(buffer)
	return EAR_mean - EAR >= EYE_CLOSE_THRESHOLD or EAR_mean < 0.14



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

	EAR_buffer_len = 50
	EAR_buffer = [0.3] * EAR_buffer_len
	eyeIsClosed = False
	eyeClosedStartTime = 0
	eyesClosedInterval = 0
	prev_nose_y = 0
	nod_counter = 0
	total_nod = 0
	show_alert = False
	EAR_buffer_idx = 0

	
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
		if len(faces) > 1:
			cv2.putText(frame, "Multiple faces detected. Only driver's face should be in the frame.",
				(100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (74, 95, 255), 2)
		elif len(faces) == 1:
			face = faces[0]
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
			if EAR_buffer_idx < EAR_buffer_len:
				EAR_buffer[EAR_buffer_idx] = EAR_AVG
				EAR_buffer_idx += 1
			else:
				eyeIsClosed = eyeClosed(EAR_AVG, EAR_buffer)
				if eyeIsClosed and eyeClosedStartTime == 0:
					eyeClosedStartTime = time.time()
				elif not eyeIsClosed and eyeClosedStartTime > 0:
					eyesClosedInterval = time.time() - eyeClosedStartTime
					print("==> Eyes closed interval: {:.3f}".format(eyesClosedInterval))
					eyeClosedStartTime = 0
			
			if input_args.verbose:
				face_bb = face_utils.rect_to_bb(face)
				face_rect = cv2.rectangle(frame, (face_bb[0], face_bb[1]), 
					(face_bb[0] + face_bb[2], face_bb[1] + face_bb[3]), (62, 255, 132), 2)
				cv2.putText(face_rect, 'Face', 
					(face_bb[0], face_bb[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (62, 255, 132))
				cv2.putText(frame, "Nodding Cnt: {}".format(total_nod), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
				cv2.putText(frame, "EAR (AVG): {:.2f}".format(EAR_AVG), (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		else:
			cv2.putText(frame, "No face detected.",
				(360, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q") or key == ord("Q"):
			break
		elif key == ord("p") or key ==  ord("P"):
			pause = True
		cv2.imshow("Driver Drowsiness Detector", frame)

	print("==> Detector closing...")

	cv2.destroyAllWindows()
	stream.stop()