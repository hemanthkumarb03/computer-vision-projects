import cv2 as cv
import mediapipe as mp
import time 
from  mediapipe.python.solutions.hands import Hands 
from mediapipe.python.solutions import drawing_utils 
import mediapipe.python.solutions.hands as mphands





class handetector:
    def __init__(self,mode=False, maxHands = 2,dconf=0.5,tconf=0.5):
        self.mode = mode 
        self.maxHands = maxHands 
        self.dconf = dconf 
        self.tconf = tconf 
        self.hands = Hands(self.mode, self.maxHands, 1 ,self.dconf, self.tconf)
        self.mpdraw = drawing_utils
    
    def findHands(self,frame,draw=True):
        frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        #rint(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for eachframe in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(frame,eachframe, mphands.HAND_CONNECTIONS)
            
        return frame 
    
    def findpos(self,frame,handno=0,draw=True):
        lmlist = [] 
        if self.results.multi_hand_landmarks:
            myhands = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhands.landmark):
                #print(id,lm)
                h,w,c = frame.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx,cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(frame,(cx,cy),15,(255,0,255),cv.FILLED)

        return lmlist 
    


def main():
    ptime=0
    ctime=0
    cap = cv.VideoCapture(0)
    detector = handetector()
    while True:
        success, frame = cap.read()
        frame = detector.findHands(frame=frame)
        lmlist = detector.findpos(frame)
        if len(lmlist) != 0:
            print(lmlist[4])
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime 
        cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv.imshow("video",frame)
        cv.waitKey(1)




if __name__ == "__main__":
    main()