import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from train_rcnn import YoloDetectionDataset, collate_fn, T
import argparse
from typing import Dict, List, Tuple

CLASS_NAMES = ["step", "stair", "grab_bar", "ramp"]

def load_model(
    model_path: str, num_classes: int = 5, device: str = "cuda"
) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def calculate_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    total_predictions = 0
    total_targets = 0
    true_positives = 0

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]

        total_predictions += len(pred_boxes)
        total_targets += len(gt_boxes)

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)

        max_ious, max_gt_idx = ious.max(dim=1)

        for pred_idx, (iou, gt_idx) in enumerate(zip(max_ious, max_gt_idx)):
            if iou > iou_threshold and pred_labels[pred_idx] == gt_labels[gt_idx]:
                true_positives += 1

    precision = true_positives / total_predictions if total_predictions > 0 else 0.0
    recall = true_positives / total_targets if total_targets > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "total_predictions": total_predictions,
        "total_targets": total_targets,
    }


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    class_names: List[str] = None,
) -> Dict[str, float]:
    model.eval()
    all_predictions = []
    all_targets = []

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        class_metrics=True,
        max_detection_thresholds=[1, 10, 100],
    )

    print("Running evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            metric.update(predictions, targets)

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    map_results = metric.compute()

    metrics_50 = calculate_metrics(all_predictions, all_targets, iou_threshold=0.5)

    print("\nEvaluation Results:")
    print(f"mAP@50: {map_results['map_50'].item():.4f}")
    print(f"mAP@50:95: {map_results['map'].item():.4f}")
    print(f"\nPrecision@IoU=0.5: {metrics_50['precision']:.4f}")
    print(f"Recall@IoU=0.5: {metrics_50['recall']:.4f}")

    # Print per-class metrics if class names are provided
    if class_names and "map_per_class" in map_results:
        print("\nPer-class mAP@50:")
        for class_id, class_map in enumerate(map_results["map_per_class"]):
            class_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else f"Class {class_id}"
            )
            print(f"{class_name}: {class_map.item():.4f}")


    # Combine all metrics
    results = {
        "map_50": map_results["map_50"].item(),
        "map_50_95": map_results["map"].item(),
        "precision_50": metrics_50["precision"],
        "recall_50": metrics_50["recall"],
    }

    if "map_per_class" in map_results:
        results["map_per_class"] = [m.item() for m in map_results["map_per_class"]]

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="wm_barriers_data",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    args = parser.parse_args()

    dataset = YoloDetectionDataset(
        image_dir=os.path.join(args.data_dir, f"images/{args.split}"),
        label_dir=os.path.join(args.data_dir, f"labels/{args.split}"),
        transform=T.ToTensor(),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"\nEvaluating on {args.split} set...")
    print(f"Dataset size: {len(dataset)} images")
    print("Classes:", ", ".join(CLASS_NAMES))

    model = load_model(args.model_path, device=args.device)

    results = evaluate_model(model, data_loader, args.device, CLASS_NAMES)

    # Save results to file
    results_file = os.path.join(
        os.path.dirname(args.model_path), f"evaluation_results_{args.split}.txt"
    )
    with open(results_file, "w") as f:
        f.write("Evaluation Results:\n")
        f.write(f"mAP@50: {results['map_50']:.4f}\n")
        f.write(f"mAP@50:95: {results['map_50_95']:.4f}\n")
        f.write(f"Precision@IoU=0.5: {results['precision_50']:.4f}\n")
        f.write(f"Recall@IoU=0.5: {results['recall_50']:.4f}\n")

        if "map_per_class" in results:
            f.write("\nPer-class mAP@50:\n")
            for i, map_score in enumerate(results["map_per_class"]):
                f.write(f"{CLASS_NAMES[i]}: {map_score:.4f}\n")

        f.write("\nSize-based Metrics:\n")
        f.write(f"mAP (small objects): {results['map_small']:.4f}\n")
        f.write(f"mAP (medium objects): {results['map_medium']:.4f}\n")
        f.write(f"mAP (large objects): {results['map_large']:.4f}\n")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
