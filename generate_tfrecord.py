"""
Usage:
  python generate_tfrecord.py --images_path=output_dataset/train --label_map=label_map.pbtxt --output_path=train.record
  python generate_tfrecord.py --images_path=output_dataset/test --label_map=label_map.pbtxt --output_path=test.record
"""

import os
import glob
import tensorflow as tf
import argparse
from object_detection.utils import dataset_util, label_map_util
from PIL import Image
from pathlib import Path

def create_tf_example(image_path, annotations, label_map_dict):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    filename = os.path.basename(image_path).encode('utf8')
    encoded_image = img.tobytes()
    image_format = b'jpeg' if image_path.endswith('.jpg') else b'png'

    image_name = Path(image_path).stem
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    if image_name in annotations:
        for obj in annotations[image_name]:
            xmins.append(obj['xmin'] / width)
            xmaxs.append(obj['xmax'] / width)
            ymins.append(obj['ymin'] / height)
            ymaxs.append(obj['ymax'] / height)
            classes_text.append(obj['class'].encode('utf8'))
            classes.append(label_map_dict[obj['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def load_annotations(label_dir):
    annotations = {}
    for txt_file in glob.glob(f"{label_dir}/*.txt"):
        name = Path(txt_file).stem
        boxes = []
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                label, x_center, y_center, width, height = parts
                x_center, y_center, width, height = map(float, parts[1:])
                xmin = (x_center - width / 2)
                ymin = (y_center - height / 2)
                xmax = (x_center + width / 2)
                ymax = (y_center + height / 2)
                boxes.append({'class': label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
        annotations[name] = boxes
    return annotations

def main(args):
    writer = tf.io.TFRecordWriter(args.output_path)
    label_map_dict = label_map_util.get_label_map_dict(args.label_map)
    annotations = load_annotations(os.path.join(args.images_path, "labels"))

    for image_path in glob.glob(f"{args.images_path}/images/*"):
        tf_example = create_tf_example(image_path, annotations, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print(f"[INFO] TFRecord created at: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', required=True)
    parser.add_argument('--label_map', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    main(args)

