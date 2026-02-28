import cv2
import numpy as np
import glob
import os
import pandas as pd
if os.path.exists("anotation_file.pd"):
    anotation_file = pd.read_csv("anotation_file.pd")
else:
    cols = ["video_path","bottom_left","bottom_right","top_left","top_right","height","width"]
    anotation_file = pd.DataFrame(columns=cols)
    anotation_file.to_csv("anotation_file.pd")
print(anotation_file)
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 4:
            pts.append((x, y))
            print(x,y)

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
    pts = []
    gif_path = f'{video_path}'
    cap = cv2.VideoCapture(gif_path)
    if not cap.isOpened():
        print("Error: Could not open GIF.")
        exit()

    ret, frame = cap.read()
    while True:
        if not ret: break
        for p in pts:
            cv2.circle(frame, p, 5, (0, 255, 0), -1)
        cv2.imshow('Court Intersection Corners',frame)
        cv2.setMouseCallback("Court Intersection Corners", click_event)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        if len(pts) == 4:
            break
    bottom_left = pts[0]
    bottom_right = pts[1]
    top_left = pts[2]
    top_right = pts[3]
    n_pts = np.array(pts).reshape(-1)
    maxx = np.maximum(n_pts,0)
    print(maxx)
    anotation = {
        "video_path":video_path,
        "bottom_left":bottom_left,
        "bottom_right":bottom_right,
        "top_left":top_left,
        "top_right":top_right,
        "height":frame.shape[0],
        "width":frame.shape[1],   
    }
    anotation_file.loc[len(anotation_file)] = anotation
    anotation_file.to_csv("anotation_file.pd")

cap.release()
cv2.destroyAllWindows()