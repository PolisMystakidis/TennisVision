import cv2
import numpy as np
import glob
import os

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360

def get_intersection(line1, line2):
    """Calculates the intersection of two lines given by (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Determinant formula for intersection
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None  # Lines are parallel
    
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return int(px), int(py)

# 1. Setup the path to your folder
video_folder = 'videos'
# This grabs all mp4, avi, and mov files
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.gif']
video_files = []
for ext in video_extensions:
    video_files.extend(glob.glob(os.path.join(video_folder, ext)))
    
for video_path in video_files:
    gif_path = f'{video_path}'
    cap = cv2.VideoCapture(gif_path)
    end_points = []
    if not cap.isOpened():
        print("Error: Could not open GIF.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        top_down_frame = None
        frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 250, 255)
        # edges = cv2.Canny(gray, 100, 250)
        # Returns a list of lines [x1, y1, x2, y2]
        kernel_sharpen = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])
        sharpened = cv2.filter2D(frame, -1, kernel_sharpen*0.2)
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 210])  # High brightness
        upper_white = np.array([180, 45, 255]) # Low saturation (no color)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        edges = cv2.filter2D(mask,-1,kernel_sharpen*0.2)
        height = frame.shape[0]
        width = frame.shape[1]
        line_length_var = 0.2
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=50, minLineLength=min(height,width)*line_length_var, maxLineGap=5)
        if lines is not None:
            # Separate lines into Horizontal and Vertical based on slope
            horizontals = []
            verticals = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1): # More horizontal than vertical
                    horizontals.append(line[0])
                else:
                    verticals.append(line[0])

            # 3. Find Intersections (Corners)
            last_point = None
            for h in horizontals:
                for v in verticals:
                    point = get_intersection(h, v)
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.imshow('Court Intersection Corners',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()