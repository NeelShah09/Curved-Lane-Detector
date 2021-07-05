import cv2
import numpy as np
import LaneUtilities as util

video = "lane.mp4"
cap = cv2.VideoCapture(video)

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('lane_detection_2.mp4', fourcc, 25.0, (1280, 720))

left_average, right_average = np.array([]), np.array([])

while cap.isOpened():
    timer = cv2.getTickCount()
    success, frame = cap.read()
    
    if frame is not None:
        imgPerspectiveCopy = frame.copy()
        imgPoints = frame.copy()
        imgLines = frame.copy()
        lineImage = frame.copy()
        imageBlank = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        curvedLane = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        framePoints = [(550, 460), (742, 460), (128, 720), (1280, 720)]

        Mask, CannyImg = util.preprocessing(frame)
        
        ROIMask = np.zeros_like(CannyImg)
        cv2.fillPoly(ROIMask, np.array([[framePoints[0], framePoints[1], framePoints[3], framePoints[2]]]), 255)
        
        CannyImg = cv2.bitwise_and(CannyImg, ROIMask)
        
        ImageWithPoints = util.drawPoints(imgPoints, framePoints)
        perspectiveImage = util.getPerspectiveImage(Mask, framePoints)
        
        # lineImage, lineImage2 = util.detectLines(CannyImg, lineImage, frame)
        windowImage, averageFit, centreOffset = util.slidingWindow(perspectiveImage, curvedLane, windowWidth=200, drawBoxes=False, averageReadings=10)
        curvature = util.getCurvature(frame, averageFit)
        
        inversePerspective = util.getInvPerspectiveImage(windowImage, framePoints)
        imgPerspectiveCopy = cv2.addWeighted(imgPerspectiveCopy, 1, inversePerspective, 0.3, 0)
        
        cv2.putText(imgPerspectiveCopy, "Curvature : "+curvature+' m', (0, int(frame.shape[0]*0.1)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(imgPerspectiveCopy, "Centre Offset : "+str(centreOffset)+' m', (0, int(frame.shape[0]*0.15)), cv2.FONT_HERSHEY_DUPLEX , 1, (0, 0, 0), 2, cv2.LINE_AA)
        fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
        cv2.putText(imgPerspectiveCopy, "FPS : " + str(int(fps)), (0, int(frame.shape[0] * 0.20)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # stackedImages = util.stackImages([[frame, CannyImg, ImageWithPoints], [perspectiveImage, lineImage, lineImage2], [imgPerspectiveCopy, windowImage, Mask]], (640, 360))
        out.write(imgPerspectiveCopy)
        cv2.imshow("Result", cv2.resize(imgPerspectiveCopy, None, fx=1.5, fy=1.5))
        # cv2.imshow("Stacked", stackedImages)
        key = cv2.waitKey(1)
        if key == 27:
            out.release()
            webcam.release()
            break
        elif key == ord('s'):
            cv2.imwrite(imgPerspectiveCopy, r"D:\Python Projects\OpenCV\Lane Detection\Sample Output.jpg")
    else:
        break
out.release()
webcam.release()
cv2.destroyAllWindows()
