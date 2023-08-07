import datetime
import os
import winsound
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import tkinter

page = tkinter.Tk()
page.title("Driver Drowsiness Monitoring")
# page.geometry("550x400")
page.geometry("{}x{}".format(page.winfo_screenwidth(), page.winfo_screenheight()))

font = ('times', 17, 'bold')
# title = tkinter.Label(page, text='Driver Drowsiness Detection',anchor=tkinter.W, justify=tkinter.CENTER)
title = tkinter.Label(page, text='Driver Drowsiness Detection', anchor='center', justify='center')
title.pack(fill='both', expand=True)

title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


def EAR(drivereye):
    point1 = dist.euclidean(drivereye[1], drivereye[5])
    point2 = dist.euclidean(drivereye[2], drivereye[4])
    # computing the euclidean distance between the horizontal
    distance = dist.euclidean(drivereye[0], drivereye[3])
    # computing the eye aspect ratio
    eye_aspect_ratio = (point1 + point2) / (2.0 * distance)
    return eye_aspect_ratio

def MOR(drivermouth):
    # computing the euclidean distances between the horizontal
    point   = dist.euclidean(drivermouth[0], drivermouth[6])
    # computing the euclidean distances between the vertical
    point1  = dist.euclidean(drivermouth[2], drivermouth[10])
    point2  = dist.euclidean(drivermouth[4], drivermouth[8])
    # taking average
    Ypoint   = (point1+point2)/2.0
    # computing mouth aspect ratio
    mouth_aspect_ratio = Ypoint/point
    return mouth_aspect_ratio

def createLog(message):
    # current_time = datetime.datetime.now()
    current_time = datetime.datetime.now()
    date_str = current_time.strftime("%d-%m-%y")
    time_str = current_time.strftime("%H:%M:%S")
    if os.path.exists("log.xlsx") == True:
        data = pd.read_excel("log.xlsx")
        data = data.values
        data = np.vstack([data, [message, str(date_str), str(time_str)]])
    else:
        data = []
        data.append([message, str(date_str), str(time_str)])
        data = np.asarray(data)
    df = pd.DataFrame(data, columns=['Alert','Date','Time'])
    df.to_excel("log.xlsx", index=False)

def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def startDetection():
    webcamera = cv2.VideoCapture(0)
    svm_predictor_path = 'SVMclassifier.dat'


    svm_detector = dlib.get_frontal_face_detector()
    svm_predictor = dlib.shape_predictor(svm_predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    eyeThreshValue = 0.25
    eyeRelativeFrames = 10
    mouthThreshValue = 0.75

    frameCount = 0 

    yawnStatus = False

    yawnCount = 0

    while True:
        ret, frame = webcamera.read()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = svm_detector(gray, 0)
        recentYawnStatus = yawnStatus
        for rect in rects:
            shape = svm_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            mar = MOR(mouth)
            
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if ear < eyeThreshValue:
                frameCount += 1
                cv2.putText(frame, "Eyes Closed ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if frameCount >= eyeRelativeFrames:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    createLog("Eyes Closed Alert")
                    beep()
            else:
                frameCount = 0
                cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if mar > mouthThreshValue:
                cv2.putText(frame, "Yawning, DROWSINESS ALERT! ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawnStatus = True
                output_text = "Yawn Count: " + str(yawnCount + 1)
                cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
                createLog("Yawning Alert")
                beep()
            else:
                yawnStatus = False
            if recentYawnStatus == True and yawnStatus == False:
                yawnCount+=1
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    webcamera.release()    




font1 = ('times', 15, 'bold')
button = tkinter.Button(page, text="Start Detecting", command=startDetection)
# button.place(x=50,y=200)
button.place(relx=0.5, rely=0.5, anchor='center')
button.config(font=font1)  

pathlabel = tkinter.Label(page)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=250)


page.config(bg='chocolate1')
page.mainloop()