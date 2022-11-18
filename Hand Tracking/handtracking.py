import cv2 as cv
import mediapipe as mp
import time 
from  mediapipe.python.solutions.hands import Hands 
from mediapipe.python.solutions import drawing_utils 
import mediapipe.python.solutions.hands as mphands

cap = cv.VideoCapture(0)
hands = Hands()
mpdraw = drawing_utils
ptime=0
ctime=0
while True:
    success, frame = cap.read()

    frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    #rint(results.multi_hand_landmarks)
    for eachframe in results.multi_hand_landmarks:
        for id, lm in enumerate(eachframe.landmark):
            #print(id,lm)
            h,w,c = frame.shape
            cx,cy = int(lm.x * w), int(lm.y * h)
            print(id, cx,cy)
            if id==4:
                cv.circle(frame,(cx,cy),15,(255,0,255),cv.FILLED)
        mpdraw.draw_landmarks(frame,eachframe, mphands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime 
    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv.imshow("video",frame)
    cv.waitKey(1)

