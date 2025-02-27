import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.utils.data import DataLoader, Dataset
from glob import glob
import json
from tqdm import tqdm

np.random.seed(3)

def load_cityscapes_train_data(image_dir, ann_dir):
    """
    Load training data for fine-tuning from Cityscapes dataset.
    """
    image_paths = glob(os.path.join(image_dir, '*', '*_leftImg8bit.png'))
    annotation_paths = [
        path.replace("leftImg8bit", "gtFine").replace("_leftImg8bit.png", "_gtFine_polygons.json")
        for path in image_paths
    ]
    return image_paths, annotation_paths

def collate_fn(batch):
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)

class CityscapesDataset(Dataset):
    def __init__(self, image_paths, annotation_paths, transform=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        with open(self.annotation_paths[idx], 'r') as f:
            data = json.load(f)

        mask = Image.new('L', image.size, 0)
        for obj in data['objects']:
            polygon = [(int(x), int(y)) for x, y in obj['polygon']]
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)

        if self.transform:
            image, mask = self.transform(image, mask)

        return torch.tensor(np.array(image) / 255.0).permute(2, 0, 1), torch.tensor(np.array(mask))

def finetune_model(model, train_loader, num_epochs=5, lr=1e-4, device="cuda"):
    """
    Fine-tune the model using Cityscapes training data.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.float())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Paths for Cityscapes dataset
train_image_dir = os.path.join('cityscapes_dataset', 'leftImg8bit', 'train')
train_ann_dir = os.path.join('cityscapes_dataset', 'gtFine', 'train')

# Load training data
train_image_paths, train_annotation_paths = load_cityscapes_train_data(train_image_dir, train_ann_dir)

# Create dataset and dataloader
dataset = CityscapesDataset(train_image_paths, train_annotation_paths)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Load model
checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
model = build_sam2(model_cfg, checkpoint).to("cuda")

# Fine-tune model
finetune_model(model, train_loader, num_epochs=5, lr=1e-4, device="cuda")

# Save fine-tuned model
fine_tuned_checkpoint = "checkpoints/sam2.1_hiera_base_plus_finetuned.pt"
torch.save(model.state_dict(), fine_tuned_checkpoint)
print(f"Fine-tuned model saved to {fine_tuned_checkpoint}")

# Prediction process (as in original code)
# Predictor setup
predictor = SAM2ImagePredictor(model)

test_image_dir = os.path.join('cityscapes_dataset', 'leftImg8bit', 'val')
image_paths = glob(os.path.join(test_image_dir, '*', '*_leftImg8bit.png'))
iou_results_all = []

for img_path in tqdm(image_paths):
    image = Image.open(img_path)
    predictor.set_image(image)

    ann_path = img_path.replace('_leftImg8bit.png', '_gtFine_polygons.json').replace('leftImg8bit', 'gtFine')
    with open(ann_path, 'r') as f:
        data = json.load(f)

    input_boxes = []
    for obj in data['objects']:
        polygon = obj['polygon']
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
        input_boxes.append([x_min, y_min, x_max, y_max])

    input_boxes = np.array(input_boxes)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # IoU calculation and result saving (same as original code)
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
with open('./cityscapes_dataset/results_b+.json', 'w') as f:
    json.dump(iou_results_all, f)