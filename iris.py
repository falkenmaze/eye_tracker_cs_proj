import cv2 as cv 
import mediapipe as mp 
import numpy as np
mpface_mesh = mp.solutions.face_mesh
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
cap = cv.VideoCapture(0)
with mpface_mesh.FaceMesh(
		max_num_faces=1,
		refine_landmarks = True,
		min_detection_confidence = 0.7,
		min_tracking_confidence = 0.7
	) as face:
	while True:
		ret, frame = cap.read()
		frame = cv.flip(frame, 1)
		frame_conv = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		results = face.process(frame_conv)
		if results.multi_face_landmarks:
			img_height, img_width = frame.shape[:2]
			points = np.array([np.multiply([p.x,p.y], [img_width, img_height]) for p in results.multi_face_landmarks[0].landmark])
			(l_cx, l_cy), l_radius = cv.minEnclosingCircle(points[LEFT_IRIS])
			(r_cx, r_cy), r_radius = cv.minEnclosingCircle(points[RIGHT_IRIS])
			center_left = np.array([l_cx, l_cy], dtype=np.int32)
			center_right = np.array([r_cx, r_cy], dtype=np.int32)
			cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
			cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)
		cv.imshow('video', frame)
		if cv.waitKey(1)  & 0xFF == ord('q'):
			break

cap.release()
cv.destroyAllWindows()
