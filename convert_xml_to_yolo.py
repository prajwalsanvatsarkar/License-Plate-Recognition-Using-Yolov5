import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    for file in os.listdir(xml_dir):
        if not file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        yolo_lines = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name != 'licence':
                continue  # Only convert licence class

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        txt_filename = file.replace('.xml', '.txt')
        txt_path = os.path.join(label_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    print(f"✅ Converted XML in {xml_dir} → YOLO format in {label_dir}")

# === Run for both train and test sets ===
convert_voc_to_yolo(
    xml_dir='./output_dataset(1)/output_dataset/train/labels_xml',
    label_dir='./output_dataset(1)/output_dataset/train/labels'
)

convert_voc_to_yolo(
    xml_dir='./output_dataset(1)/output_dataset/test/labels_xml',
    label_dir='./output_dataset(1)/output_dataset/test/labels'
)

