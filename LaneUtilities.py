import cv2
import numpy as np

left_a, left_b, left_c, right_a, right_b, right_c = [], [], [], [], [], []

def preprocessing(src):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lower_yellow, upper_yellow = np.array([18, 54, 160]), np.array([48, 255, 255])
    lower_white, upper_white = np.array([0, 0, 210]), np.array([255, 255, 255])
    
    yellowMask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    whiteMask = cv2.inRange(hsv, lower_white, upper_white)
    
    combined = cv2.bitwise_or(yellowMask, whiteMask)
    
    kernal = np.ones((1, 1))
    imgDil = cv2.dilate(combined, kernal, iterations=3)
    
    imgCanny = cv2.Canny(imgDil, 255, 255, apertureSize=7)
    imgCanny = cv2.dilate(imgCanny, kernal, iterations=1)
    return imgDil, imgCanny

def drawPoints(src, pointsList):
    for point in pointsList:
        cv2.circle(src, point, 5, (0, 0, 255), 5)
    return src

def cvt1Ch23Ch(src):
    return cv2.merge((src, src, src))

def getPerspectiveImage(src, points):
    W = src.shape[1]
    H = src.shape[0]
    warpPoints = np.float32([(0, 0), (W, 0), (0, H), (W, H)])
    points = np.float32(points)
    matrix = cv2.getPerspectiveTransform(points, warpPoints)
    warped = cv2.warpPerspective(src, matrix, (W, H))
    return warped

def getInvPerspectiveImage(src, points):
    W = src.shape[1]
    H = src.shape[0]
    invwarpPoints = np.float32([(0, 0), (W, 0), (0, H), (W, H)])
    points = np.float32(points)
    invMatrix = cv2.getPerspectiveTransform(invwarpPoints, points)
    return cv2.warpPerspective(src, invMatrix, (W, H))

def getHistogram(src, scale=0.5):
    return np.sum(src[int((1-scale)*src.shape[0]):, :], axis=0, keepdims=True)

def slidingWindow(image, hist3Ch, nwindows=15, windowWidth=100, minPix=1, histScale=0.5, drawBoxes=False, averageReadings=10):
    global left_a, left_b, left_c, right_a, right_b, right_c
    width, height = image.shape[1], image.shape[0]
    
    #Plot Histogram and find start of lanes in left and right halves of image
    hist = getHistogram(image, histScale)
    midpoint = image.shape[1]//2
    leftPeak = np.argmax(hist[:, :midpoint])
    rightPeak = np.argmax(hist[:, midpoint:])+midpoint
    centreOffset = round(((width-(leftPeak+rightPeak))/2)*3/900, 3)
    
    #Find points that have non-zero value i.e. white patches in black image
    NonZero_Points = image.nonzero()
    X_Coordinates_Of_Non_Zero_Points = NonZero_Points[1]
    Y_Coordinates_Of_Non_Zero_Points = NonZero_Points[0]
    
    #Draw boxes and find good points coordinates
    current_left, current_right = leftPeak, rightPeak
    left_lane_points, right_lane_points = [], []
    
    for i in range(nwindows):
        left_x_top = current_left-windowWidth//2
        left_x_bottom = current_left+windowWidth//2
        right_x_top = current_right-windowWidth//2
        right_x_bottom = current_right+windowWidth//2
        y_upper = int(height*(nwindows-i-1)/nwindows)
        y_lower = int(height*(nwindows-i)/nwindows)
        if drawBoxes:
            cv2.rectangle(hist3Ch, (left_x_top, y_upper), (left_x_bottom, y_lower), (0, 255, 255), 1)
            cv2.rectangle(hist3Ch, (right_x_top, y_upper), (right_x_bottom, y_lower), (0, 255, 255), 1)
        
        Index_Of_Point_In_Left_Box = ((X_Coordinates_Of_Non_Zero_Points > left_x_top) & 
                                     (X_Coordinates_Of_Non_Zero_Points < left_x_bottom) &
                                     (Y_Coordinates_Of_Non_Zero_Points > y_upper) &
                                     (Y_Coordinates_Of_Non_Zero_Points < y_lower)).nonzero()[0]
        
        Index_Of_Point_In_Right_Box = ((X_Coordinates_Of_Non_Zero_Points > right_x_top) & 
                                      (X_Coordinates_Of_Non_Zero_Points < right_x_bottom) &
                                      (Y_Coordinates_Of_Non_Zero_Points > y_upper) &
                                      (Y_Coordinates_Of_Non_Zero_Points < y_lower)).nonzero()[0]
        
        #Update current centre of left and right boxes
        if len(Index_Of_Point_In_Left_Box) > minPix:
            current_left = int(np.mean(X_Coordinates_Of_Non_Zero_Points[Index_Of_Point_In_Left_Box]))
        if len(Index_Of_Point_In_Right_Box) > minPix:
            current_right = int(np.mean(X_Coordinates_Of_Non_Zero_Points[Index_Of_Point_In_Right_Box]))
            
        left_lane_points.append(Index_Of_Point_In_Left_Box)
        right_lane_points.append(Index_Of_Point_In_Right_Box)
    
    left_lane_points = np.concatenate(left_lane_points)
    right_lane_points = np.concatenate(right_lane_points)
    
    X_Coordinates_Of_Left_Lane_Point = X_Coordinates_Of_Non_Zero_Points[left_lane_points]
    Y_Coordinates_Of_Left_Lane_Point = Y_Coordinates_Of_Non_Zero_Points[left_lane_points]
    X_Coordinates_Of_Right_Lane_Point = X_Coordinates_Of_Non_Zero_Points[right_lane_points]
    Y_Coordinates_Of_Right_Lane_Point = Y_Coordinates_Of_Non_Zero_Points[right_lane_points]
    
    left_fit = np.polyfit(Y_Coordinates_Of_Left_Lane_Point, X_Coordinates_Of_Left_Lane_Point, 2)
    right_fit = np.polyfit(Y_Coordinates_Of_Right_Lane_Point, X_Coordinates_Of_Right_Lane_Point, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    la = np.mean(left_a[-averageReadings:])
    lb = np.mean(left_b[-averageReadings:])
    lc = np.mean(left_c[-averageReadings:])
    ra = np.mean(right_a[-averageReadings:])
    rb = np.mean(right_b[-averageReadings:])
    rc = np.mean(right_c[-averageReadings:])
    
    curve_fit_y_points = np.arange(0, height+1)
    curve_fit_y_points_reversed = np.arange(height, -1, -1)
    left_curve_fit_x_points = la*(curve_fit_y_points**2) + lb*curve_fit_y_points + lc
    right_curve_fit_x_points = ra*(curve_fit_y_points_reversed**2) + rb*curve_fit_y_points_reversed + rc
    average_fit = [(la+ra)/2, (lb+rb)/2, (lc+rc)/2]
    Y = np.append(curve_fit_y_points, curve_fit_y_points_reversed)
    X = np.append(left_curve_fit_x_points, right_curve_fit_x_points)
    
    poly_points = np.array(list(zip(np.int64(X),Y)))
    cv2.fillPoly(hist3Ch, [poly_points], (0, 255, 0))
    return hist3Ch, average_fit, centreOffset

def getCurvature(image, average_fit, threshold=6000):
    threshold = abs(threshold)
    X = image.shape[0]
    curve = pow(1+(2*average_fit[0]*(X-100) + average_fit[1])**2, 1.5)/(2*average_fit[0])
    if curve > threshold or curve < -threshold:
        return 'inf'
    return str(round(curve, 2))

def stackImages(imgArray, size, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], size)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank]*rows
        hor_con = [imgBlank]*rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], size)
            if len(imgArray[x].shape)==2:
                imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver
