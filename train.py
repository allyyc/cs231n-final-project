from ultralytics import YOLO
import argparse
import comet_ml

def main(args):
    model = YOLO(args.model)

    # Build training arguments with only user-provided augmentations
    train_args = {
        "data": "wm_barriers_data/data.yaml",
        "epochs": 100,
        "batch": 16,
        "device": 0,
    }

    # Only add augmentation args if the user explicitly set them
    if args.hsv_h is not None:
        train_args["hsv_h"] = args.hsv_h
    if args.hsv_s is not None:
        train_args["hsv_s"] = args.hsv_s
    if args.hsv_v is not None:
        train_args["hsv_v"] = args.hsv_v
    if args.translate is not None:
        train_args["translate"] = args.translate
    if args.scale is not None:
        train_args["scale"] = args.scale
    if args.fliplr is not None:
        train_args["fliplr"] = args.fliplr
    if args.mosaic is not None:
        train_args["mosaic"] = args.mosaic
    if args.erasing is not None:
        train_args["erasing"] = args.erasing
    if args.auto_augment is not None:
        train_args["auto_augment"] = args.auto_augment
    if args.degrees is not None:
        train_args["degrees"] = args.degrees
    if args.mixup is not None:
        train_args["mixup"] = args.mixup
    if args.shear is not None:
        train_args["shear"] = args.shear
    if args.perspective is not None:
        train_args["perspective"] = args.perspective
    if args.cutmix is not None:
        train_args["cutmix"] = args.cutmix
    if args.mixup is not None:
        train_args["mixup"] = args.mixup



    model.train(**train_args)

if __name__ == "__main__":
    comet_ml.init()
    parser = argparse.ArgumentParser(description="Train YOLO on accessibility barriers with optional augmentations")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--hsv_h", type=float, default=None)
    parser.add_argument("--hsv_s", type=float, default=None)
    parser.add_argument("--hsv_v", type=float, default=None)
    parser.add_argument("--translate", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--fliplr", type=float, default=None)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--erasing", type=float, default=None).
    parser.add_argument("--auto_augment", type=str, default=None)
    parser.add_argument("--degrees", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--shear", type=float, default=None)
    parser.add_argument("--perspective", type=float, default=None)
    parser.add_argument("--cutmix", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)

    args = parser.parse_args()
    main(args)
