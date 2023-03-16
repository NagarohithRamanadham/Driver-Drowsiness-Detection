
from imutils import face_utils
import imutils
import dlib
import cv2


    

webcamera = cv2.VideoCapture(0)
svm_predictor_path = 'SVMclassifier.dat'


svm_detector = dlib.get_frontal_face_detector()
svm_predictor = dlib.shape_predictor(svm_predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
while True:
    ret, frame = webcamera.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = svm_detector(gray, 0)
    for rect in rects:
        shape = svm_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
webcamera.release()    

