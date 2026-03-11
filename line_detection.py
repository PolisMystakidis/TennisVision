import cv2
import numpy as np
import glob
import torch
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from post_process import postprocess, refine_kps , postprocess_ball
from homography import get_trans_matrix, refer_kps
from models import CourtTrackerNet , BallTrackerNet
from tqdm import tqdm
from scipy.spatial import distance

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360
use_homography = True
court_model_dir = "D:\\jupyter_server\\TennisVision\\model_tennis_court_det.pt"
ball_model_dir = "D:\\jupyter_server\\TennisVision\\model_ball.pt"
image_dir = "D:\\jupyter_server\\datasets\\data\\images"
image_id = "-_5ljBK4HnI_200.png"
out_path = "D:\\jupyter_server\\TennisVision\\outputs"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device : ",device)

model = CourtTrackerNet(out_channels=15)
model.load_state_dict(torch.load(court_model_dir, map_location=device))
model = model.to(device)

ball_model = BallTrackerNet()
ball_model.load_state_dict(torch.load(ball_model_dir, map_location=device))
ball_model = ball_model.to(device)

def detect_lines(image,model,use_refine_kps=True,use_homography=True):
    inp = (image.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0))
    inp = inp.unsqueeze(0)
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
            points = [np.squeeze(x).tolist() for x in points]
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

def infer_model(frames, model,scaleH = None ,scaleW = None):
    """ Run pretrained model on a consecutive list of frames    
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return    
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """
    height = OUTPUT_HEIGHT
    width = OUTPUT_WIDTH
    dists = [-1]*2
    ball_track = [(None,None)]*2
    for num in range(2, len(frames)):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num-1], (width, height))
        img_preprev = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)
        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess_ball(output,OUTPUT_HEIGHT,OUTPUT_WIDTH)
        ball_track.append((x_pred, y_pred))
        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)
    return ball_track, dists 

def remove_outliers(ball_track, dists, max_dist = 10):
    """ Remove outliers from model prediction    
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if i+1 < len(dists) and i-1 >= 0 :
            if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
                ball_track[i] = (None, None)
                outliers.remove(i)
            elif dists[i-1] == -1:
                ball_track[i-1] = (None, None)
    return ball_track  

if __name__ == "__main__":
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
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        last_3_frames = []
        frame_id = 0
        avrg_court_points = []
        if not cap.isOpened():
            print("Error: Could not open GIF.")
            exit()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            last_3_frames.append(frame)
            last_3_frames = last_3_frames[-3::]
            
            #Ball Detection
            ball_track , dist = infer_model(last_3_frames,ball_model)
            ball_track = remove_outliers(ball_track, dist)

            # Court Detection
            # Detect line for first 5 frames then take avrg 
            if len(avrg_court_points) < 5:
                court_points = detect_lines(frame,model,use_refine_kps=True,use_homography=False)
                if (None,None) not in court_points:
                    avrg_court_points.append(court_points)
            elif len(avrg_court_points) >= 5:
                court_points = np.asarray(avrg_court_points).sum(axis=0)/5
                court_points = court_points.tolist()
            #Homography
            if (None,None) not in court_points:
                if use_homography:
                    refer_kps = np.float32([[100,100],[100,OUTPUT_HEIGHT-100],[OUTPUT_WIDTH-100,100],[OUTPUT_WIDTH-100,OUTPUT_HEIGHT-100]]).reshape(4,2)
                    original_points = np.float32([court_points[4],court_points[5],court_points[6],court_points[7]]).reshape(4,2)
                    matrix = cv2.getPerspectiveTransform(original_points,refer_kps)
                    image = cv2.warpPerspective(image, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            
            # Draw on image
            image = frame
            # image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype = np.uint8)
            if ball_track[-1][0] is not None and ball_track[-1][1] is not None:  
                x_ball_pred , y_ball_pred = ball_track[-1]
                image = cv2.circle(image, (int(x_ball_pred), int(y_ball_pred)),
                                        radius=0, color=(255,0, 0), thickness=10)
            for j in range(len(court_points)):
                if court_points[j][0] is not None:
                    image = cv2.circle(image, (int(court_points[j][0]), int(court_points[j][1])),
                                radius=0, color=(0,255, 0), thickness=10)
            



            cv2.imshow('Court Intersection Corners',image)
            frame_id += 1
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()