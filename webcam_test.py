import cv2
import numpy as np
from line_detection import infer_model,remove_outliers,ball_model,OUTPUT_HEIGHT,OUTPUT_WIDTH

cam = cv2.VideoCapture(0)
last_3_frames = []
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    image = frame
    last_3_frames.append(frame)
    last_3_frames = last_3_frames[-3::]
    #Ball Detection
    if len(last_3_frames) == 3:
        ball_track , dist = infer_model(last_3_frames,ball_model)
        ball_track = remove_outliers(ball_track, dist)
        #Draw ball 
        if ball_track[-1][0] is not None and ball_track[-1][1] is not None:
            print("detected")  
            x_ball_pred , y_ball_pred = ball_track[-1]
            image = cv2.circle(image, (int(x_ball_pred), int(y_ball_pred)),
                                    radius=0, color=(255,0, 0), thickness=10)
    # Display the captured frame
    cv2.imshow('Camera', image)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()