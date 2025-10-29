'''
Script to generate Surgical Tooltip Annotations for the JIGSAWS dataset.
Uses SAM2 to segment the tool based on bounding box prompts and extract the tooltip based on a fixed condition (top-left or top-right most pixel).
'''

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import json
from tqdm import tqdm

video_path = '../JIGSAWS/Suturing/video/'
box_path = './box_annotations'
video_list = [
    f for f in os.listdir(video_path) 
    if f.endswith('.avi') and os.path.exists(os.path.join(box_path, f.replace('.avi', '.json')))
]
print("num videos:", len(video_list))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import hydra
from omegaconf import OmegaConf

# Add the model_cfg to Hydra's config search path
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path="../sam2/sam2", job_name="sam2_job")
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_base_plus.pt"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda:0')

predictor = SAM2ImagePredictor(sam2_model)

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def test_tool_position(box, image):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    if euclidean_distance(x2, y2, w, h) < euclidean_distance(x1, y2, 0, h):
        return 1 #box represents right tool
    return 0 #else box represents left tool

def smallest_bounding_box(boxes):
    #Compute the area of each box and find the one with the smallest area
    return min(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))

# 
def find_tooltip_coordinates(mask, label):
    #Find the coordinates of the non-zero pixels in the mask
    coords = np.column_stack(np.where(mask == 1))

    if label == 'L':
        #For the left tool, find the right-most point (largest x-value)
        right_most_point = coords[np.argmax(coords[:, 1])]
        return (right_most_point[1].item(), right_most_point[0].item())  # Return (x, y)

    else:
        #For the right tool, find the left-most point (smallest x-value)
        left_most_point = coords[np.argmin(coords[:, 1])]
        return (left_most_point[1].item(), left_most_point[0].item())  # Return (x, y)

def main():
    for video_name in tqdm(video_list, position=0, desc="Processing Videos"):
        with open(os.path.join(box_path, video_name.replace('.avi', '.json')), 'r') as f:
            box_annotation = json.load(f)
        
        cap = cv2.VideoCapture(os.path.join(video_path, video_name))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        segment_annotations = {}
        ann_frame_idx = 0
        pbar = tqdm(total=length, desc=f'Processing {video_name}', position=1, leave=False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, :, ::-1].copy() #bgr -> RGB
            predictor.set_image(frame)
            box_data = box_annotation.get(str(ann_frame_idx), {})
            
            if box_data:
                frame_annotation = {'objects': []}
                bboxes = box_data['bboxes']
                rtool_boxes, ltool_boxes = [], []
                for box in bboxes:
                    if test_tool_position(box, frame):
                        rtool_boxes.append(box)
                    else:
                        ltool_boxes.append(box)
                #If there are multiple boxes, we select the tightest box 
                if len(rtool_boxes) > 1:
                    rtool_boxes = [smallest_bounding_box(rtool_boxes)]
                if len(ltool_boxes) > 1:
                    ltool_boxes = [smallest_bounding_box(ltool_boxes)]
                
                lbox = list(map((lambda x: int(x)), ltool_boxes[0])) if len(ltool_boxes) > 0 else None
                rbox = list(map((lambda x: int(x)), rtool_boxes[0])) if len(rtool_boxes) > 0 else None
                tool_labels = []
                if lbox and rbox:
                    box_prompts = np.array([lbox, rbox])
                    tool_labels = ['L', 'R']
                elif lbox:
                    box_prompts = np.array([lbox])
                    tool_labels = ['L']
                else:
                    box_prompts = np.array([rbox])
                    tool_labels = ['R']
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_prompts,
                    multimask_output=False,
                )
                if len(masks.shape) == 3:
                    masks = masks[np.newaxis, :]
                    
                for obj_idx in range(len(masks)):
                    mask = masks[obj_idx, 0, :, :]
                    mask = (mask > 0.0).astype(np.uint8)
        
                    # coords = np.column_stack(np.where(mask == 1))
                    # coords = [(int(x), int(y)) for y, x in coords]
                    
                    tooltip = find_tooltip_coordinates(mask, tool_labels[obj_idx])
        
                    annot = {
                        #'mask': coords,
                        'tooltip': tooltip
                    }
                    
                    frame_annotation['objects'].append(annot)
        
                frame_annotation['labels'] = tool_labels
                segment_annotations[ann_frame_idx] = frame_annotation

            ann_frame_idx += 1
            pbar.update(1)
        
        json_name = video_name.replace('.avi', '.json')
        os.makedirs('./segment_annotations/', exist_ok=True)
        with open('./segment_annotations/'+json_name, 'w+') as f:
            json.dump(segment_annotations, f)

if __name__ == "__main__":
    main()