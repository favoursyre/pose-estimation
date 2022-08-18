#I want to create a generic Pose estimation module using opencv and mediapipe framework

#Useful libraries for working with Computer Vision -->
#Python libraries 
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import time
import math

#Computer Vision libraries
import cv2 as cv
import mediapipe as mp



print("POSE ESTIMATION MODULE \n")

class poseDetector():
    #Declaring the various arguments
    def __init__(self, staticImageMode = False, complexity = 1, upBody = False, 
                 smoothLm = False, minDetectionConfidence = 0.7, minTrackingConfidence = 0.7): #The mode is false because we want it to alternate between tracking and detecting, it only detects if its set to True 
        self.staticImageMode = staticImageMode
        self.complexity = complexity
        self.upBody = upBody
        self.smoothLm = smoothLm
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        #We are instantiating the objects of hand tracking functions
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(staticImageMode, complexity, upBody, 
                                        smoothLm, minDetectionConfidence, minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    #This function handles the various pose landmarks in the frame
    def findPose(self, image, draw = True):
        global results
        rgbImage = cv.cvtColor(image, cv.COLOR_BGR2RGB) #We are converting the format because mediapipe works with RGB format
        results = self.pose.process(rgbImage) #This gets the various points in the hands
        #print(f"Results: {results}")
        #print(f"Result landmarks: {results.multi_hand_landmarks} \n")

        if results.pose_landmarks:
            poseLms = results.pose_landmarks
            #for poseLms in results.pose_world_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, poseLms, self.mpPose.POSE_CONNECTIONS) #This gets the drawings in the hands
            else:
                pass
        else:
            pass
        return image

    #This function allows us to find all positions in the body
    def findPosition(self, image, poseNo = [10, 12, 13], draw = True): #, handNo = 8
        global yList, xList, bbox, lmList
        yList = []
        xList = []
        bbox = []
        lmList = []
        if results.pose_landmarks:
            #for hands in results.multi_hand_landmarks: #[handNo]
            for id, lm in enumerate(results.pose_landmarks.landmark): #.landmark
                #if poseNo: #This checks if poseNo exist
                #    for i in poseNo: 
                #        if id == i:  
                h, w, c = image.shape
                #print(f"Landmark: {id}: {lm} -------------------------------->")
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(f"Landmarks: {id}: {cx} {cy} \n")
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy]) #This appends the various points to the landmark list
                        
                #This draws the circle on the selected landmark
                if draw:
                    cv.circle(image, (cx, cy), 10, (255, 0, 0), cv.FILLED)

            #This helps us get the dimensions of the bounding box for the detected hands
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            bbox = xMin, yMin, xMax, yMax
            #print(f"BBOX: {bbox}")
            if draw:
                cv.rectangle(image, (xMin - 20, yMin - 20), (xMax + 20, yMax + 20), (0, 255, 0), 2)

        #print(f"X List: {xList}")
        #print(f"Length: {len(lmList)}, Landmark List: {lmList}")
        return lmList, bbox

    #This function will help us find the angle between three landmarks in the body
    def findAngle(self, image, point1, point2, point3, draw = True):
        #This helps get the points of the various landmarks in the body
        x1, y1 = lmList[point1][1:]
        x2, y2 = lmList[point2][1:]
        x3, y3 = lmList[point3][1:]

        #This calculates the angles
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360
        #print(f"{angle} degrees")

        #This helps to draw the lines and points of the calculated angle
        if draw:
            cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv.line(image, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv.circle(image, (x1, y1), 10, (0, 0, 255), cv.FILLED)
            cv.circle(image, (x1, y1), 15, (0, 0, 255), 2)
            cv.circle(image, (x2, y2), 10, (0, 0, 255), cv.FILLED)
            cv.circle(image, (x2, y2), 15, (0, 0, 255), 2)
            cv.circle(image, (x3, y3), 10, (0, 0, 255), cv.FILLED)
            cv.circle(image, (x3, y3), 15, (0, 0, 255), 2)
            cv.putText(image, f"{int(360 - angle)} deg", (x2 - 50, y2 + 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return angle



def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture("bicep_curls.mp4")
    detector = poseDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = detector.findPose(frame)
            lmList, bbox = detector.findPosition(frame, draw = True)
            #for i in range(10):
            if len(lmList) != 0:
                #pass
                cv.circle(frame, (lmList[14][1], lmList[14][2]), 10, (255, 0, 0), cv.FILLED)
                #print(f"Landmark List: {lmList} \n")
                #print(f"1st Landmark List: {lmList[0]} \n")

            #Getting the frames per second
            cTime = time.perf_counter()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv.putText(frame, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()