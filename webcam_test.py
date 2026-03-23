import cv2
import numpy as np
from line_detection import infer_model,remove_outliers,ball_model,OUTPUT_HEIGHT,OUTPUT_WIDTH

prevCicle = None
dist = lambda x1,y1,x2,y2: (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)
GREEN_RANGE = ((30, 85, 10), (65, 255, 200)) # this is for HSV recommended
# GREEN_RANGE = ((8, 150, 110), (145, 255, 255)) # this is for RGB not recomended
def ballTrackHough(image):
    global prevCicle , dist
    kernel = np.ones((5, 5), np.uint8)
    colorLower, colorUpper = GREEN_RANGE
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grayFrame = cv2.inRange(hsv, colorLower, colorUpper)
    # grayFrame = cv2.morphologyEx(grayFrame, cv2.MORPH_CLOSE, kernel)
    # grayFrame = cv2.cvtColor(image_masked,cv2.COLOR_BGR2GRAY)
    # grayFrame = cv2.GaussianBlur(grayFrame,(9,9),2)
    circles = cv2.HoughCircles(grayFrame , cv2.HOUGH_GRADIENT , 1 , 500, param1 = 200 , param2 = 25,
                              minRadius = 5 , maxRadius = 100)
    print(circles)
    chosen = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for x in circles[0]:
            if chosen is None:chosen = x
            if prevCicle is not None:
                if dist(chosen[0],chosen[1],prevCicle[0],prevCicle[1]) >= dist(x[0],x[1],prevCicle[0],prevCicle[1]):
                    chosen = x
    return chosen , grayFrame
cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("videos\\1video_infer copy.gif")
last_3_frames = []
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    image = frame
    last_3_frames.append(frame)
    last_3_frames = last_3_frames[-3::]
    green_image = frame
    #Ball Detection
    if len(last_3_frames) == 3:
        #Deep ball Detection
        ball_track , dist = infer_model(last_3_frames,ball_model)
        ball_track = remove_outliers(ball_track, dist)
        #Traditional Ball detection
        ball , green_image = ballTrackHough(frame)
        print("Ball : ",ball)
        if ball is not None:
            ball_track[-1] = ball[0:2]
            #Draw ball 
            if ball_track[-1][0] is not None and ball_track[-1][1] is not None:
                x_ball_pred , y_ball_pred = ball_track[-1]
                image = cv2.circle(image, (int(x_ball_pred), int(y_ball_pred)),
                                        radius=0, color=(255,0, 0), thickness=10)
    # Display the captured frame
    cv2.imshow('Camera', green_image)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()