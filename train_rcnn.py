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

from torchmetrics.detection import MeanAveragePrecision

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

        if self.transform:
            image = self.transform(image)
        target = {"boxes": boxes, "labels": labels}

        return image, target


# Define the training, validation, and test datasets
train_dataset = YoloDetectionDataset(
    image_dir="wm_barriers_data/images/train",
    label_dir="wm_barriers_data/labels/train",
    transform=transforms.ToTensor(),
)
val_dataset = YoloDetectionDataset(
    image_dir="wm_barriers_data/images/val",
    label_dir="wm_barriers_data/labels/val",
    transform=transforms.ToTensor(),
)
test_dataset = YoloDetectionDataset(
    image_dir="wm_barriers_data/images/test",
    label_dir="wm_barriers_data/labels/test",
    transform=transforms.ToTensor(),
)


def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


# Define the training, validation, and test data loaders
train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn
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
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_class_loss = 0.0
    train_box_loss = 0.0

    # Create progress bar for training
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images, targets in train_pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
        train_class_loss += loss_dict["loss_classifier"].item()
        train_box_loss += loss_dict["loss_box_reg"].item()

        # Update progress bar with current loss
        train_pbar.set_postfix(
            {
                "loss": f"{losses.item():.4f}",
                "class_loss": f'{loss_dict["loss_classifier"].item():.4f}',
                "box_loss": f'{loss_dict["loss_box_reg"].item():.4f}',
            }
        )

    lr_scheduler.step()

    # Average the losses
    avg_train_loss = train_loss / len(train_loader)
    avg_train_class_loss = train_class_loss / len(train_loader)
    avg_train_box_loss = train_box_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_class_loss = 0.0
    val_box_loss = 0.0

    # Create progress bar for validation
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
    with torch.no_grad():
        for images, targets in val_pbar:
            # Calculate loss and mAP
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
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
        f"Train - Loss: {avg_train_loss:.4f}, Class Loss: {avg_train_class_loss:.4f}, Box Loss: {avg_train_box_loss:.4f}"
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
