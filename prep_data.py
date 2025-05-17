import xml.etree.ElementTree as ET
import yaml
import os


def convert_cvat_to_yolo(xml_path):
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize YOLO format dict
    yolo_data = {
        "path": "/Users/allisoncasasola/accessibility-barriers/wm_barriers_data", #  Dataset root directory
        "train": "wm_barriers_data/images",
        "val": "wm_barriers_data/images",
        "test": "wm_barriers_data/images",
        "nc": 4,  # Number of classes
        "names": ["step", "stair", "grab_bar", "ramp"],  # Class names
    }

    # Create .txt files
    for image in root.findall("image"):
        img_path = image.attrib["name"]
        img_id = image.attrib["id"]
        width = float(image.attrib["width"])
        height = float(image.attrib["height"])

        # Create label file for this image
        label_path = os.path.join(
            "wm_barriers_data",
            img_path.replace("images/", "labels/").replace(".jpg", ".txt"),
        )
        os.makedirs(
            os.path.dirname(label_path), exist_ok=True
        )  # Create labels directory if it doesn't exist

        with open(label_path, "w") as lf:
            for box in image.findall("box"):
                # Get class id
                label = box.attrib["label"]
                class_id = yolo_data["names"].index(label)

                # Convert bbox to YOLO format
                x_min = float(box.attrib["xtl"])
                y_min = float(box.attrib["ytl"])
                x_max = float(box.attrib["xbr"])
                y_max = float(box.attrib["ybr"])

                # Convert to normalized coordinates
                x_center = ((x_min + x_max) / 2) / width
                y_center = ((y_min + y_max) / 2) / height
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height

                # Write YOLO format line
                lf.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

        print(f"Created label file: {label_path}")

    # Save YAML config
    yaml_path = os.path.join("wm_barriers_data", "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yolo_data, f, sort_keys=False)
    print(f"Created YAML config: {yaml_path}")


if __name__ == "__main__":
    # Convert the annotations
    xml_path = os.path.join("wm_barriers_data", "wm_annotations.xml")
    convert_cvat_to_yolo(xml_path)
