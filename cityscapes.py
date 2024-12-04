import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import json
from glob import glob
from tqdm import tqdm
from pycocotools import mask as maskUtils

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Get all image file paths in the validation set
image_paths = glob(os.path.join('cityscapes_dataset', 'leftImg8bit', 'val', '*', '*_leftImg8bit.png'))
i = 0
iou_results_all = []

# Just for testing, process only the first 5 images
for img_path in tqdm(image_paths):
    # i += 1
    # if i > 5:
    #     break
    # Load image
    image = Image.open(img_path)
    predictor.set_image(image)
    
    # Construct the corresponding annotation file path
    ann_path = img_path.replace('_leftImg8bit.png', '_gtFine_polygons.json').replace('leftImg8bit', 'gtFine')
    
    # Load annotations
    with open(ann_path, 'r') as f:
        data = json.load(f)
    
    # Extract polygons and convert them to bounding boxes
    input_boxes = []
    for obj in data['objects']:
        # if obj['label'] in ['sky', 'road']:
        #     continue
        polygon = obj['polygon']
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        input_boxes.append([x_min, y_min, x_max, y_max])
    
    input_boxes = np.array(input_boxes)
    
    # Run prediction
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    num_masks, batch_size, H, W = masks.shape
    # Initialize a list to store IoU results
    iou_results = []

    # Iterate over each ground truth object and corresponding predicted mask
    for idx, obj in enumerate(data['objects']):
        # Create ground truth mask from polygon
        gt_polygon = [(int(x), int(y)) for x, y in obj['polygon']]
        gt_mask = Image.new('L', image.size, 0)
        ImageDraw.Draw(gt_mask).polygon(gt_polygon, outline=1, fill=1)
        gt_mask = np.array(gt_mask).astype(bool)

        # Get the predicted mask for this object
        pred_mask = masks[idx, 0].astype(bool)

        # Calculate Intersection over Union (IoU)
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = intersection / union if union != 0 else 0

        # Store the IoU result
        iou_result = {
            'image_id': img_path,
            'object_id': obj,
            'iou': iou
        }
        # print(iou_result)
        iou_results.append(iou_result)
    iou_results_all.extend(iou_results)

# Save IoU results to a JSON file
with open('./cityscapes_dataset/results_s.json', 'w') as f:
    json.dump(iou_results_all, f)

    # Optionally, save or display the results
    # For example, save the masks as a PNG image
    # result_dir = os.path.join('cityscapes_dataset', 'result_visual', 'hiera_large', 'val')
    # mask_path = os.path.join(result_dir, os.path.basename(img_path).replace('.png', '_masks.png'))
    
    # Visualize and save the results using the show_masks function
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.squeeze(0), plt.gca(), random_color=True)
    # plt.axis('off')
    # plt.show()
    # plt.savefig(mask_path)

# # After processing all images, save the IoU results to a JSON file
# result_json_path = os.path.join('cityscapes_dataset', 'result_visual', 'hiera_large', 'hiera_large.json')
# with open(result_json_path, 'w') as f:
#     json.dump(iou_results, f)

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# for mask in masks:
#     show_mask(mask.squeeze(0), plt.gca(), random_color=True)
# for box in input_boxes:
#     show_box(box, plt.gca())
# plt.axis('off')
# plt.show()

# input_point = np.array([[500, 375]])
# input_label = np.array([1])
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
# sorted_ind = np.argsort(scores)[::-1]
# masks = masks[sorted_ind]
# scores = scores[sorted_ind]
# logits = logits[sorted_ind]
# input_point = np.array([[500, 375], [1125, 625]])
# input_label = np.array([1, 0])

# mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

# masks, scores, _ = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     mask_input=mask_input[None, :, :],
#     multimask_output=True,
# )

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()
# show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)