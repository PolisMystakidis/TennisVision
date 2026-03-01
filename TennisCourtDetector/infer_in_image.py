import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
import argparse
model_dir = "D:\\jupyter_server\\TennisVision\\model_tennis_court_det.pt"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--input_path', type=str, help='path to input image')
    parser.add_argument('--output_path', type=str, help='path to output image')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    args = parser.parse_args()
    

    model = BallTrackerNet(out_channels=15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.eval()
    
    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360
    image_dir = "D:\\jupyter_server\\datasets\\data\\images"
    image_id = "-_5ljBK4HnI_200.png"
    out_path = "D:\\jupyter_server\\TennisVision\\outputs"
    
    image = cv2.imread(f"{image_dir}\\{image_id}")
    img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    inp = (img.astype(np.float32) / 255.)
    cv2.imshow('Court Intersection Corners',inp)
    cv2.waitKey()
    inp = torch.tensor(np.rollaxis(inp, 2, 0))
    inp = inp.unsqueeze(0)

    out = model(inp.float().to(device))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()
    points = []
    for kps_num in range(14):
        heatmap = (pred[kps_num]*255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if args.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    if args.use_homography:
        matrix_trans = get_trans_matrix(points)
        if matrix_trans is not None:
            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            points = [np.squeeze(x) for x in points]

    for j in range(len(points)):
        if points[j][0] is not None:
            image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                               radius=0, color=(0, 0, 255), thickness=10)
        
    cv2.imshow('Court Intersection Corners',image)
    cv2.waitKey()
    cv2.imwrite(out_path, image)
