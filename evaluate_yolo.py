import argparse
from ultralytics import YOLO

def main(model_path):
    model = YOLO(model_path)

    results = model.val(data="wm_barriers_data/data.yaml")

    print("Class indices with average precision:", results.ap_class_index)
    print("Average precision for all classes:", results.box.all_ap)
    print("Average precision:", results.box.ap)
    print("Class indices for average precision:", results.box.ap_class_index)
    print("Class-specific results:", results.box.class_result)
    print("Mean average precision:", results.box.map)
    print("Mean average precision at IoU=0.50:", results.box.map50)
    print("Mean average precision for different IoU thresholds:", results.box.maps)
    print("Mean results for different metrics:", results.box.mean_results)
    print("Mean precision:", results.box.mp)
    print("Mean recall:", results.box.mr)
    print("Precision:", results.box.p)
    print("Precision values:", results.box.prec_values)
    print("Specific precision metrics:", results.box.px)
    print("Recall:", results.box.r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model.")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO .pt model file")
    args = parser.parse_args()
    main(args.model_path)
