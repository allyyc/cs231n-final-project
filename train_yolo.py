from ultralytics import YOLO
import argparse
import comet_ml
import os


def main(args):
    # Load model from checkpoint if specified, otherwise use base model
    if args.resume and os.path.exists(args.resume):
        print(f"Loading model from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Starting training with base model: {args.model}")
        model = YOLO(args.model)

    # Build training arguments with only user-provided augmentations
    train_args = {
        "data": "wm_barriers_data/data.yaml",
        "epochs": args.epochs,  # Use epochs from command line
        "batch": args.batch_size,  # Use batch size from command line
        "device": args.device,
        "resume": (
            True if args.resume else False
        ),  # Enable resume if checkpoint provided
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

    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Model: {args.resume if args.resume else args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("\nAugmentations:")
    for key, value in train_args.items():
        if (
            key not in ["data", "epochs", "batch", "device", "resume"]
            and value is not None
        ):
            print(f"{key}: {value}")

    # Start training
    model.train(**train_args)


if __name__ == "__main__":
    comet_ml.init()
    parser = argparse.ArgumentParser(
        description="Train YOLO on accessibility barriers with optional augmentations"
    )

    # Model and training arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model to use (if not resuming)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device to train on (GPU ID or -1 for CPU)",
    )

    # Augmentation arguments
    parser.add_argument(
        "--hsv_h", type=float, default=None, help="HSV-Hue augmentation"
    )
    parser.add_argument(
        "--hsv_s", type=float, default=None, help="HSV-Saturation augmentation"
    )
    parser.add_argument(
        "--hsv_v", type=float, default=None, help="HSV-Value augmentation"
    )
    parser.add_argument(
        "--translate", type=float, default=None, help="Translation augmentation"
    )
    parser.add_argument("--scale", type=float, default=None, help="Scale augmentation")
    parser.add_argument(
        "--fliplr", type=float, default=None, help="Horizontal flip probability"
    )
    parser.add_argument(
        "--mosaic", type=float, default=None, help="Mosaic augmentation probability"
    )
    parser.add_argument(
        "--erasing", type=float, default=None, help="Random erasing probability"
    )
    parser.add_argument(
        "--auto_augment", type=str, default=None, help="Auto augmentation policy"
    )
    parser.add_argument("--degrees", type=float, default=None, help="Rotation degrees")
    parser.add_argument(
        "--mixup", type=float, default=None, help="Mixup augmentation probability"
    )
    parser.add_argument("--shear", type=float, default=None, help="Shear augmentation")
    parser.add_argument(
        "--perspective", type=float, default=None, help="Perspective augmentation"
    )
    parser.add_argument(
        "--cutmix", type=float, default=None, help="CutMix augmentation probability"
    )

    args = parser.parse_args()
    main(args)
