from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame  
import time

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	EAR = (A + B) / (2.0 * C)
	return EAR
	
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture('http://192.168.116.112:4747/mjpegfeed?640x480')
flag=0
alarm_playing = False
eyes_open_time = None

pygame.mixer.init()

try:
    while True:
        ret, frame=cap.read()
        if frame is None:
            print("Frame is None")
            continue
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            EAR = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if EAR < thresh:
                flag += 1
                print(flag)
                if flag >= frame_check:
                    print("****************ALERT!****************")
                    if not alarm_playing:
                        pygame.mixer.music.load('alarm.mp3')
                        pygame.mixer.music.play(-1)  # Play the sound in a loop
                        alarm_playing = True
            else:
                if alarm_playing:
                    if eyes_open_time is None:
                        eyes_open_time = time.time()
                    elif time.time() - eyes_open_time > 3:  # If eyes have been open for more than 3 seconds
                        pygame.mixer.music.stop()
                        alarm_playing = False
                        flag = 0
                        eyes_open_time = None
                else:
                    eyes_open_time = None
            if alarm_playing:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10,325),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    cv2.destroyAllWindows()
    cap.release()
    pygame.mixer.quit()
    print("Application closed")
