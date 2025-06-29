import argparse
from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import cv2
from torchvision.transforms import functional as F
from torch import nn
from torch.nn import functional as F_nn

from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

from torchmetrics.detection import MeanAveragePrecision

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def calculate_precision(detections, targets, iou_threshold=0.5):
    total_precision = 0.0
    num_images = len(detections)

    for det, tgt in zip(detections, targets):
        if len(det["boxes"]) == 0:
            continue

        pred_boxes = det["boxes"]
        gt_boxes = tgt["boxes"]

        if len(gt_boxes) == 0:
            continue

        ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)

        max_ious, _ = ious.max(dim=1)

        true_positives = (max_ious > iou_threshold).sum().item()

        precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
        total_precision += precision

    return total_precision / num_images if num_images > 0 else 0.0


# Define the YoloDetectionDataset class
class YoloDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        )
        self.label_files = sorted(
            [
                f.replace(".jpg", ".txt")
                .replace(".jpeg", ".txt")
                .replace(".png", ".txt")
                for f in self.image_files
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __load_labels(self, label_path, img_width, img_height):
        boxes = []
        labels = []

        try:
            with open(label_path, "r") as f:
                for line in f:
                    label, x, y, w, h = line.strip().split()
                    x = float(x) * img_width
                    y = float(y) * img_height
                    w = float(w) * img_width
                    h = float(h) * img_height
                    # Convert from [x, y, w, h] to [x1, y1, x2, y2] format
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    boxes.append([x1, y1, x2, y2])
                    labels.append(
                        int(label) + 1
                    )  # Add 1 to make labels 1-indexed (0 is background)
        except FileNotFoundError:
            print(f"Warning: Label file {label_path} not found")

        if len(boxes) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.int64
        )

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        boxes, labels = self.__load_labels(label_path, img_width, img_height)
        target = {"boxes": boxes, "labels": labels}

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    

    
def eval_forward(model, images, targets):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            bad_boxes = boxes[:, 2:] <= boxes[:, :2]
            if bad_boxes.any():
                bb_idx = torch.where(bad_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training = True

    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [
        s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
    ]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(
        objectness, pred_bbox_deltas
    )

    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(
        proposals, objectness, images.image_sizes, num_anchors_per_level
    )

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = (
        model.roi_heads.select_training_samples(proposals, targets)
    )
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(
        class_logits, box_regression, labels, regression_targets
    )
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(
        class_logits, box_regression, proposals, image_shapes
    )
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
    model.rpn.training = False
    model.roi_heads.training = False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


def load_model(model_path=None, num_classes=5):
    # Initialize the model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the head of the model with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load state dict if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    else:
        print("Initializing new model")

    return model


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Faster R-CNN model")
    parser.add_argument(
        "--load_model", type=str, help="Path to load model from (optional)"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="faster_rcnn_resnet50_fp_.pth",
        help="Path to save model to (default: faster_rcnn_resnet50_fpn.pth)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=25,
        help="Number of epochs to train (default: 25)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training (default: 2)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        help="Learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--epoch_interval",
        type=int,
        default=5,
        help="Interval to evaluate model (default: 5)",
    )

    args = parser.parse_args()

    # Initialize the experiment
    experiment = Experiment(api_key=os.getenv("COMET_API_KEY"))

    num_classes = 5

    # Load or initialize the model
    model = load_model(args.load_model, num_classes)

    train_transform = T.Compose(
        [
            # Color augmentations (HSV-like)
            T.ColorJitter(
                brightness=0.4,  # value
                contrast=0.7,  # saturation-like
                saturation=0.7,  # saturation
                hue=0.015,  # hue
            ),
            # Geometric augmentations
            T.RandomAffine(
                degrees=10.0,  # rotation
                translate=(
                    0.1,
                    0.1,
                ),  # translation range from -10% to +10% in both x and y
                scale=(0.5, 1.5),  # scale
                shear=0.0,  # shear
                fill=0,  # fill color for areas outside the image
            ),
            # Flips
            T.RandomHorizontalFlip(p=0.5),  # horizontal flip
            T.RandomVerticalFlip(p=0.0),  # vertical flip
            # Convert to tensor
            T.ToTensor(),
        ]
    )

    val_transform = T.ToTensor()

    train_dataset = YoloDetectionDataset(
        image_dir="wm_barriers_data/images/train",
        label_dir="wm_barriers_data/labels/train",
        transform=train_transform,
    )

    val_dataset = YoloDetectionDataset(
        image_dir="wm_barriers_data/images/val",
        label_dir="wm_barriers_data/labels/val",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005
    )

    metric = MeanAveragePrecision()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_loss_objectness = 0.0
        train_loss_rpn_box_reg = 0.0

        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, targets in train_pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get model predictions (returns a dict of losses)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Accumulate losses
            train_loss += losses.item()
            train_loss_objectness += loss_dict["loss_objectness"].item()
            train_loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()

            # Update progress bar
            train_pbar.set_postfix(
                {
                    "loss": f"{losses.item():.4f}",
                    "cls": f'{loss_dict["loss_classifier"].item():.4f}',
                    "box": f'{loss_dict["loss_box_reg"].item():.4f}',
                }
            )

        lr_scheduler.step()

        # Validation loop
        if epoch % args.epoch_interval == 0:
            model.eval()
            val_loss = 0.0
            val_loss_objectness = 0.0
            val_loss_rpn_box_reg = 0.0

            # Initialize metric with specific settings
            metric = MeanAveragePrecision(
                box_format="xyxy",
                iou_thresholds=[
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                class_metrics=True,
                max_detection_thresholds=[
                    1,
                    10,
                    100,
                ],
            )

            # Create progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            with torch.no_grad():
                for images, targets in val_pbar:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    # Get predictions and losses
                    loss_dict, detections = eval_forward(model, images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    # Update validation losses
                    val_loss += losses.item()
                    val_loss_objectness += loss_dict["loss_objectness"].item()
                    val_loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()

                    # Update mAP metric
                    metric.update(detections, targets)

                    # Update progress bar
                    val_pbar.set_postfix(
                        {
                            "loss": f"{losses.item():.4f}",
                            "cls": f'{loss_dict["loss_classifier"].item():.4f}',
                            "box": f'{loss_dict["loss_box_reg"].item():.4f}',
                            "objectness": f'{loss_dict["loss_objectness"].item():.4f}',
                            "rpn_box_reg": f'{loss_dict["loss_rpn_box_reg"].item():.4f}',
                        }
                    )

            # Calculate mAP@50 and other metrics
            map_results = metric.compute()
            map_50 = map_results["map_50"].item()
            map_50_95 = map_results["map"].item()  # This is mAP@50:95

            # Calculate precision at different IoU thresholds
            precision_50 = calculate_precision(detections, targets, iou_threshold=0.5)
            precision_75 = calculate_precision(detections, targets, iou_threshold=0.75)

            # Print detailed metrics
            print("\nDetailed Metrics:")
            print(f"mAP@50: {map_50:.4f}")
            print(f"mAP@50:95: {map_50_95:.4f}")
            print(f"mAP@75: {map_results['map_75'].item():.4f}")
            print("\nPrecision at different IoU thresholds:")
            print(f"Precision@IoU=0.5: {precision_50:.4f}")
            print(f"Precision@IoU=0.75: {precision_75:.4f}")
            print(f"Objectness Loss: {val_loss_objectness / len(val_loader):.4f}")
            print(f"RPN Box Reg Loss: {val_loss_rpn_box_reg / len(val_loader):.4f}")

            if "map_per_class" in map_results:
                print("\nPer-class mAP@50:")
                for class_id, class_map in enumerate(map_results["map_per_class"]):
                    print(f"Class {class_id}: {class_map.item():.4f}")

            print(
                f"Val   - Avg Loss: {val_loss / len(val_loader):.4f}, "
            )

        # Print epoch train summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(
            f"Train - Avg Loss: {train_loss / len(train_loader):.4f}, "
            f"Objectness Loss: {train_loss_objectness / len(train_loader):.4f}, "
            f"RPN Box Reg Loss: {train_loss_rpn_box_reg / len(train_loader):.4f}"
        )

        # Log metrics
        experiment.log_metric("mAP@50", map_50, step=epoch)
        experiment.log_metric("mAP@50:95", map_50_95, step=epoch)
        experiment.log_metric("mAP@75", map_results["map_75"].item(), step=epoch)
        experiment.log_metric("precision@50", precision_50, step=epoch)
        experiment.log_metric("precision@75", precision_75, step=epoch)
        experiment.log_metric("map_small", map_results["map_small"].item(), step=epoch)
        experiment.log_metric(
            "map_medium", map_results["map_medium"].item(), step=epoch
        )
        experiment.log_metric("map_large", map_results["map_large"].item(), step=epoch)
        experiment.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
        experiment.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
        experiment.log_metric(
            "train_loss_objectness",
            train_loss_objectness / len(train_loader),
            step=epoch,
        )
        experiment.log_metric(
            "train_loss_rpn_box_reg",
            train_loss_rpn_box_reg / len(train_loader),
            step=epoch,
        )
        experiment.log_metric(
            "val_loss_objectness",
            val_loss_objectness / len(val_loader),
            step=epoch,
        )
        experiment.log_metric(
            "val_loss_rpn_box_reg",
            val_loss_rpn_box_reg / len(val_loader),
            step=epoch,
        )

    # Save the model using the specified path
    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved to {args.save_model}")

if __name__ == "__main__":
    main()
