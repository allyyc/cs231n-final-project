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

from torchmetrics.detection import MeanAveragePrecision


class BoxLoss(nn.Module):
    def __init__(self, beta=1.0):
        """
        Smooth L1 Loss (Huber Loss) for box regression.

        Args:
            beta (float): The threshold where the loss changes from L1 to L2.
                        Default is 1.0 as in the original Fast R-CNN paper.
        """
        super().__init__()
        self.beta = beta

    def forward(self, pred_boxes, target_boxes):
        """
        Compute the smooth L1 loss between predicted and target boxes.

        Args:
            pred_boxes (Tensor): Predicted boxes of shape [N, 4]
            target_boxes (Tensor): Target boxes of shape [N, 4]

        Returns:
            Tensor: The smooth L1 loss
        """
        # Ensure boxes are in the same format
        assert pred_boxes.shape == target_boxes.shape, "Box shapes must match"

        # Compute absolute difference
        abs_diff = torch.abs(pred_boxes - target_boxes)

        # Compute smooth L1 loss
        smooth_l1 = torch.where(
            abs_diff < self.beta,
            0.5 * abs_diff**2 / self.beta,
            abs_diff - 0.5 * self.beta,
        )

        return smooth_l1.mean()


class ClassificationLoss(nn.Module):
    def __init__(self, num_classes, weight=None):
        """
        Cross Entropy Loss for classification with optional class weights.

        Args:
            num_classes (int): Number of classes (including background)
            weight (Tensor, optional): Class weights to handle class imbalance
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, pred_logits, target_labels):
        """
        Compute the classification loss.

        Args:
            pred_logits (Tensor): Predicted class logits of shape [N, num_classes]
            target_labels (Tensor): Target class labels of shape [N]

        Returns:
            Tensor: The classification loss
        """
        # Ensure inputs are in the correct format
        assert (
            pred_logits.shape[1] == self.num_classes
        ), f"Expected {self.num_classes} classes"
        assert pred_logits.shape[0] == target_labels.shape[0], "Batch sizes must match"

        # Compute cross entropy loss
        loss = self.ce_loss(pred_logits, target_labels)

        # Average over valid samples (ignore background class if needed)
        valid_mask = target_labels >= 0  # -1 is often used as ignore label
        if valid_mask.any():
            loss = loss[valid_mask].mean()
        else:
            loss = loss.mean()

        return loss


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, box_beta=1.0, class_weight=None):
        """
        Combined loss for object detection.

        Args:
            num_classes (int): Number of classes (including background)
            box_beta (float): Beta parameter for smooth L1 loss
            class_weight (Tensor, optional): Class weights for classification
        """
        super().__init__()
        self.box_loss = BoxLoss(beta=box_beta)
        self.cls_loss = ClassificationLoss(num_classes, weight=class_weight)

    def forward(self, predictions, targets):
        """
        Compute the combined detection loss.

        Args:
            predictions (Dict): Model predictions containing:
                - boxes (Tensor): Predicted boxes [N, 4]
                - labels (Tensor): Predicted class logits [N, num_classes]
            targets (Dict): Ground truth containing:
                - boxes (Tensor): Target boxes [N, 4]
                - labels (Tensor): Target class labels [N]

        Returns:
            Dict: Dictionary containing:
                - loss_box_reg: Box regression loss
                - loss_classifier: Classification loss
                - total_loss: Combined loss
        """
        # Extract predictions and targets
        pred_boxes = predictions["boxes"]
        pred_logits = predictions["labels"]
        target_boxes = targets["boxes"]
        target_labels = targets["labels"]

        # Compute losses
        box_loss = self.box_loss(pred_boxes, target_boxes)
        cls_loss = self.cls_loss(pred_logits, target_labels)

        # Combine losses (you can adjust the weights)
        total_loss = box_loss + cls_loss

        return {
            "loss_box_reg": box_loss,
            "loss_classifier": cls_loss,
            "total_loss": total_loss,
        }


# Initialize the experiment
experiment = Experiment(api_key=os.getenv("COMET_API_KEY"))

# Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Log the model
experiment.log_model("faster_rcnn_resnet50_fpn", model)

# Number of classes (your dataset classes + 1 for background)
num_classes = 5

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the head of the model with a new one (for the number of classes in your dataset)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Initialize the custom loss
detection_loss = DetectionLoss(num_classes=num_classes)


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

        # If no boxes were found, add a dummy box
        if len(boxes) == 0:
            # Add a small box in the corner with background class (0)
            boxes = [[0, 0, 10, 10]]
            labels = [0]  # Background class

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
            if isinstance(self.transform, YOLOAugmentation):
                image, target = self.transform(image, target)
            else:
                image = self.transform(image)

        return image, target


class YOLOAugmentation:
    def __init__(
        self,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
    ):
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.fliplr = fliplr
        self.mosaic = mosaic
        self.mixup = mixup
        self.cutmix = cutmix

    def __call__(self, image, target):
        # Convert PIL image to numpy array for OpenCV operations
        img = np.array(image)

        # HSV augmentation
        if random.random() < 0.5:
            r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
            dtype = img.dtype
            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            img_hsv = cv2.merge(
                (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
            )
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        # Convert back to PIL for torchvision transforms
        image = Image.fromarray(img)

        # Random rotation
        if self.degrees > 0:
            angle = random.uniform(-self.degrees, self.degrees)
            image, target = self._rotate_image_and_boxes(image, target, angle)

        # Random translation
        if self.translate > 0:
            translate_x = random.uniform(-self.translate, self.translate)
            translate_y = random.uniform(-self.translate, self.translate)
            image, target = self._translate_image_and_boxes(
                image, target, translate_x, translate_y
            )

        # Random scale
        if self.scale > 0:
            scale = random.uniform(1 - self.scale, 1 + self.scale)
            image, target = self._scale_image_and_boxes(image, target, scale)

        # Random shear
        if self.shear > 0:
            shear_x = random.uniform(-self.shear, self.shear)
            shear_y = random.uniform(-self.shear, self.shear)
            image, target = self._shear_image_and_boxes(image, target, shear_x, shear_y)

        # Random perspective
        if self.perspective > 0:
            image, target = self._perspective_transform(image, target)

        # Random flips
        if random.random() < self.flipud:
            image, target = self._flip_vertical(image, target)
        if random.random() < self.fliplr:
            image, target = self._flip_horizontal(image, target)

        # Convert to tensor
        image = F.to_tensor(image)

        return image, target

    def _rotate_image_and_boxes(self, image, target, angle):
        # Get image dimensions
        w, h = image.size
        # Rotate image
        image = F.rotate(image, angle)
        # Rotate boxes
        boxes = target["boxes"]
        if len(boxes) > 0:
            # Convert boxes to center format
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]

            # Rotate centers
            angle_rad = angle * np.pi / 180
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Calculate new centers
            cx_new = cx * cos_a - cy * sin_a + w / 2 * (1 - cos_a) + h / 2 * sin_a
            cy_new = cx * sin_a + cy * cos_a + h / 2 * (1 - cos_a) - w / 2 * sin_a

            # Convert back to box format and ensure minimum size
            boxes[:, 0] = torch.clamp(cx_new - width / 2, 0, w - 1)
            boxes[:, 1] = torch.clamp(cy_new - height / 2, 0, h - 1)
            boxes[:, 2] = torch.clamp(cx_new + width / 2, 1, w)
            boxes[:, 3] = torch.clamp(cy_new + height / 2, 1, h)

            # Ensure minimum box size
            min_size = 1.0
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            mask = (width < min_size) | (height < min_size)
            if mask.any():
                # For boxes that are too small, expand them
                boxes[mask, 0] = torch.clamp(
                    boxes[mask, 0] - (min_size - width[mask]) / 2, 0, w - 1
                )
                boxes[mask, 2] = torch.clamp(boxes[mask, 0] + min_size, 1, w)
                boxes[mask, 1] = torch.clamp(
                    boxes[mask, 1] - (min_size - height[mask]) / 2, 0, h - 1
                )
                boxes[mask, 3] = torch.clamp(boxes[mask, 1] + min_size, 1, h)

            target["boxes"] = boxes

        return image, target

    def _translate_image_and_boxes(self, image, target, translate_x, translate_y):
        w, h = image.size
        translate_x = int(w * translate_x)
        translate_y = int(h * translate_y)

        # Translate image
        image = F.affine(
            image, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0
        )

        # Translate boxes
        boxes = target["boxes"]
        if len(boxes) > 0:
            boxes[:, 0] = torch.clamp(boxes[:, 0] + translate_x, 0, w - 1)
            boxes[:, 1] = torch.clamp(boxes[:, 1] + translate_y, 0, h - 1)
            boxes[:, 2] = torch.clamp(boxes[:, 2] + translate_x, 1, w)
            boxes[:, 3] = torch.clamp(boxes[:, 3] + translate_y, 1, h)

            # Ensure minimum box size
            min_size = 1.0
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            mask = (width < min_size) | (height < min_size)
            if mask.any():
                # For boxes that are too small, expand them
                boxes[mask, 0] = torch.clamp(
                    boxes[mask, 0] - (min_size - width[mask]) / 2, 0, w - 1
                )
                boxes[mask, 2] = torch.clamp(boxes[mask, 0] + min_size, 1, w)
                boxes[mask, 1] = torch.clamp(
                    boxes[mask, 1] - (min_size - height[mask]) / 2, 0, h - 1
                )
                boxes[mask, 3] = torch.clamp(boxes[mask, 1] + min_size, 1, h)

            target["boxes"] = boxes

        return image, target

    def _scale_image_and_boxes(self, image, target, scale):
        w, h = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Scale image
        image = F.resize(image, (new_h, new_w))

        # Scale boxes
        boxes = target["boxes"]
        if len(boxes) > 0:
            boxes[:, 0] = torch.clamp(boxes[:, 0] * scale, 0, new_w - 1)
            boxes[:, 1] = torch.clamp(boxes[:, 1] * scale, 0, new_h - 1)
            boxes[:, 2] = torch.clamp(boxes[:, 2] * scale, 1, new_w)
            boxes[:, 3] = torch.clamp(boxes[:, 3] * scale, 1, new_h)

            # Ensure minimum box size
            min_size = 1.0
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            mask = (width < min_size) | (height < min_size)
            if mask.any():
                # For boxes that are too small, expand them
                boxes[mask, 0] = torch.clamp(
                    boxes[mask, 0] - (min_size - width[mask]) / 2, 0, new_w - 1
                )
                boxes[mask, 2] = torch.clamp(boxes[mask, 0] + min_size, 1, new_w)
                boxes[mask, 1] = torch.clamp(
                    boxes[mask, 1] - (min_size - height[mask]) / 2, 0, new_h - 1
                )
                boxes[mask, 3] = torch.clamp(boxes[mask, 1] + min_size, 1, new_h)

            target["boxes"] = boxes

        return image, target

    def _shear_image_and_boxes(self, image, target, shear_x, shear_y):
        # For now, skip shear augmentation as it's complex to maintain valid boxes
        return image, target

    def _perspective_transform(self, image, target):
        # For now, skip perspective transform as it's complex to maintain valid boxes
        return image, target

    def _flip_horizontal(self, image, target):
        # Flip image
        image = F.hflip(image)

        # Flip boxes
        boxes = target["boxes"]
        if len(boxes) > 0:
            w = image.size[0]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            # Ensure x1 < x2
            boxes[:, 0], boxes[:, 2] = (
                torch.min(boxes[:, [0, 2]], dim=1)[0],
                torch.max(boxes[:, [0, 2]], dim=1)[0],
            )
            target["boxes"] = boxes

        return image, target

    def _flip_vertical(self, image, target):
        # Flip image
        image = F.vflip(image)

        # Flip boxes
        boxes = target["boxes"]
        if len(boxes) > 0:
            h = image.size[1]
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            # Ensure y1 < y2
            boxes[:, 1], boxes[:, 3] = (
                torch.min(boxes[:, [1, 3]], dim=1)[0],
                torch.max(boxes[:, [1, 3]], dim=1)[0],
            )
            target["boxes"] = boxes

        return image, target


# Define the training, validation, and test datasets
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
            translate=(0.1, 0.1),  # translation
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

# Validation and test transforms (just convert to tensor)
val_transform = T.ToTensor()

# Define the training, validation, and test datasets
full_train_dataset = YoloDetectionDataset(
    image_dir="wm_barriers_data/images/train",
    label_dir="wm_barriers_data/labels/train",
    transform=train_transform,
)

full_val_dataset = YoloDetectionDataset(
    image_dir="wm_barriers_data/images/val",
    label_dir="wm_barriers_data/labels/val",
    transform=val_transform,
)

# Create small subsets for testing
train_dataset = torch.utils.data.Subset(full_train_dataset, range(5))
val_dataset = torch.utils.data.Subset(full_val_dataset, range(5))


def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


# Define the training, validation, and test data loaders
train_loader = DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
)

# Define the metric
metric = MeanAveragePrecision()

# Define the learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Define the training loop
num_epochs = 2  # Reduced from 100 to 2 for testing
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_class_loss = 0.0
    train_box_loss = 0.0

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
        train_class_loss += loss_dict['loss_classifier'].item()
        train_box_loss += loss_dict['loss_box_reg'].item()

        # Update progress bar
        train_pbar.set_postfix({
            "loss": f"{losses.item():.4f}",
            "cls": f"{loss_dict['loss_classifier'].item():.4f}",
            "box": f"{loss_dict['loss_box_reg'].item():.4f}"
        })

    lr_scheduler.step()

    model.eval()
    val_loss = 0.0
    val_class_loss = 0.0
    val_box_loss = 0.0

    # Create progress bar for validation
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
    with torch.no_grad():
        for images, targets in val_pbar:
            # During validation, we still want to compute losses
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # During validation, we can also get predictions if needed
            # predictions = model(images)  # This would give us the detection results
            loss_dict = model(images, targets)  # This gives us the losses
            print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())

            val_loss += losses.item()
            val_class_loss += loss_dict["loss_classifier"].item()
            val_box_loss += loss_dict["loss_box_reg"].item()

            # Update progress bar with current loss
            val_pbar.set_postfix(
                {
                    "loss": f"{losses.item():.4f}",
                    "class_loss": f'{loss_dict["loss_classifier"].item():.4f}',
                    "box_loss": f'{loss_dict["loss_box_reg"].item():.4f}',
                }
            )
    # Calculate mAP
    results = metric.compute()
    map_50 = results["map_50"]

    # Average the losses
    avg_val_loss = val_loss / len(val_loader)
    avg_val_class_loss = val_class_loss / len(val_loader)
    avg_val_box_loss = val_box_loss / len(val_loader)

    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
    print(
        f"Train - Avg Loss: {train_loss / len(train_loader):.4f}, Class Loss: {train_class_loss / len(train_loader):.4f}, Box Loss: {train_box_loss / len(train_loader):.4f}"
    )
    print(
        f"Val   - Loss: {avg_val_loss:.4f}, Class Loss: {avg_val_class_loss:.4f}, Box Loss: {avg_val_box_loss:.4f}"
    )
    print(f"mAP@50: {map_50:.4f}\n")

    experiment.log_metric("mAP@50", map_50, step=epoch)
    experiment.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
    experiment.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
    experiment.log_metric(
        "train_class_loss", train_class_loss / len(train_loader), step=epoch
    )
    experiment.log_metric(
        "val_class_loss", val_class_loss / len(val_loader), step=epoch
    )
    experiment.log_metric(
        "train_box_loss", train_box_loss / len(train_loader), step=epoch
    )
    experiment.log_metric("val_box_loss", val_box_loss / len(val_loader), step=epoch)

    # Log the model
    experiment.log_model("faster_rcnn_resnet50_fpn", model)

# Save the model
torch.save(model.state_dict(), "faster_rcnn_resnet50_fpn.pth")


# Add inference function
def predict(model, image):
    """
    Run inference on a single image.

    Args:
        model: The trained Faster R-CNN model
        image: A PIL Image or tensor of shape [C, H, W]

    Returns:
        predictions: A dictionary containing:
            - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format
            - labels (Int64Tensor[N]): the predicted labels
            - scores (Tensor[N]): the scores for each detection
    """
    model.eval()
    if isinstance(image, Image.Image):
        image = T.ToTensor()(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        predictions = model(image)

    # Remove batch dimension and move to CPU
    predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions][0]
    return predictions
