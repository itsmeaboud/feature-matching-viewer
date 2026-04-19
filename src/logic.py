import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
submodule = root / "external" / "superglue"

if str(submodule) not in sys.path:
    sys.path.insert(0, str(submodule))
import cv2
import torch
import numpy as np
from models.matching import Matching
from models.utils import frame2tensor

import matplotlib.cm as cm
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
from jaxtyping import Int, Float
from typing import List
import os
import tempfile




# Initliaze matching


def load_model():
    #Configure backend
    cfg = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    matching = Matching(cfg).eval().to(device)
    return matching

'''
----------TO DO-------------
def save_images(stiched_images,
                strips):
    
    num_images = stiched_images.shape[0]

    for idx in range(num_images):

        bgr_img = cv2.cvtColor(stiched_images)
        # Only high threshold
        lines = strips[idx]
        colors = strips[idx][1]

        for start, end in lines:

'''      




def match_image(matching: Matching,
                images_path: List[Path],
                anchor_idx: Int = 1,
                threshold: Float = 0.7) :
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    images_rgb = [cv2.imread(path, cv2.IMREAD_COLOR_RGB) for path in images_path]
    images_gray = [cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) for img_rgb in images_rgb]

    if images_gray is None:
        print("ERROE: Could not load images")
        return 
    
    num_images = len(images_gray)

    images_tensor = [frame2tensor(img, device) for img in images_gray]
    
    anchor_image_gray = images_gray.pop(anchor_idx - 1)
    anchor_image_rgb = images_rgb.pop(anchor_idx - 1)
    anchor_image_tensor = images_tensor.pop(anchor_idx - 1)


   
    stiched_images = []
    strips_all = []
    hists = []

    H0, W0 = anchor_image_gray.shape

    for idx in range(num_images - 1):

        H1, W1 = images_gray[idx].shape
        H, W = max(H0, H1) , W0 + W1
        out_image = np.zeros((H, W, 3))

        out_image[:H0, :W0] = anchor_image_rgb
        out_image[:H1, W0:] = images_rgb[idx]
        stiched_images.append(out_image)
        
        with torch.no_grad():
            pred = matching({'image0': anchor_image_tensor,
                            'image1': images_tensor[idx]})
        
        
        
        # Extract 
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        conf = pred['matching_scores0'][0].cpu().numpy()
        # Filter
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        
        color = cm.jet(mconf)
        hist_ind, bin_edges = np.histogram(mconf, bins = 10, range = (0, 1))
        hists.append(hist_ind)

        filter_1 = np.where(conf[valid] >= threshold)
        filter_2 = np.where(conf[valid] < threshold)

        shifted_mkpts1 = mkpts1.copy()
        shifted_mkpts1[:, 0]+= W0

        mkpts0_1 = mkpts0[filter_1]
        shifted_mkpts1_1  = shifted_mkpts1[filter_1]
        color1 = color[filter_1]

        mkpts0_2 = mkpts0[filter_2]
        shifted_mkpts1_2  = shifted_mkpts1[filter_2]
        color2 = color[filter_2]

        strips1 = np.stack([mkpts0_1, shifted_mkpts1_1], axis = 1)
        strips2 = np.stack([mkpts0_2, shifted_mkpts1_2], axis = 1)


        strips_all.append([strips1, color1, strips2, color2])
        print(f"Total Number of Matches is: {len(mkpts0)}")




 
    anchor_image_rgb = anchor_image_rgb / 255
    stiched_images = np.array(stiched_images) / 255


    return log_matches(stiched_images,
                strips_all,
                hists)

def log_matches(images: Float[np.ndarray, "N H W 3"],
                strips: Int[List, "N 4 Mi 2 2"],
                hist):
    
    rr.init("Feature Matching Viewer", spawn = True)
    rr.log("scene",
           rr.Clear(recursive = True)
           )
    
    frames = images.shape[0]

    for idx in range(frames):
        
        rr.set_time("idx", sequence = idx)
        rr.log("scene/images",
               rr.Image(images[idx])
            )
        
        # Matches
        rr.log("scene/images/matches1",
               rr.LineStrips2D(strips[idx][0], colors = strips[idx][1])
            )
        
        rr.log("scene/images/matches2",
               rr.LineStrips2D(strips[idx][2], colors = strips[idx][3])
            )
        rr.log("scene/stats/conf_hist",
               rr.BarChart(hist[idx])
            )
    rr.send_blueprint(create_blueprint())
    return None



def create_blueprint():

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial2DView(origin = "scene/images", name = "matches"),
            rrb.BarChartView(origin = "scene/stats")
        ),
        collapse_panels = True
    )
    

    return blueprint



if __name__ == '__main__':

    img_paths = ["assets/images/1.jpeg", "assets/images/2.jpeg", "assets/images/3.jpeg", "assets/images/4.jpeg"]
    matching = load_model()
    match_image(matching, img_paths)



