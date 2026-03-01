import cv2
import numpy as np
import glob
import torch
import os
from TennisCourtDetector.tracknet import BallTrackerNet
import torch.nn.functional as F
from TennisCourtDetector.postprocess import postprocess, refine_kps
from TennisCourtDetector.homography import get_trans_matrix, refer_kps
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360
model_dir = "D:\\jupyter_server\\TennisVision\\model_tennis_court_det.pt"
image_dir = "D:\\jupyter_server\\datasets\\data\\images"
image_id = "-_5ljBK4HnI_200.png"
out_path = "D:\\jupyter_server\\TennisVision\\outputs"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device : ",device)
def detect_lines(image,use_refine_kps=True,use_homography=True):
    inp = (image.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0))
    inp = inp.unsqueeze(0)

    model = BallTrackerNet(out_channels=15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.eval()

    out = model(inp.float().to(device))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()
    points = []
    for kps_num in range(14):
        heatmap = (pred[kps_num]*255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    if use_homography:
        matrix_trans = get_trans_matrix(points)
        if matrix_trans is not None:
            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            points = [np.squeeze(x) for x in points]
    return points
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
        frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        if not ret: break
        points = detect_lines(frame)
        for j in range(len(points)):
            if points[j][0] is not None:
                image = cv2.circle(frame, (int(points[j][0]), int(points[j][1])),
                                radius=0, color=(0, 0, 255), thickness=10)
        cv2.imshow('Court Intersection Corners',image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()