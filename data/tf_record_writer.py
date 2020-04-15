"""
Writes tf_records given a input format.
"""

import argparse
import os
from os.path import join
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from data.trackmate_xml_to_csv import extract_points_from_trackmate_xml


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def is_image(file):
    try:
        end = file.split(".")[1]
        if end in ["png", "jpg", "tif", "tiff"]:
            return True
        return False
    except:
        print("WARNING: unknown image file in image folder: ", file)


def write_tf_records(image_dir, bboxes, filename="bboxes.tfrecord"):
    """
    images: paths to image
    bboxes: list of bboxes in (x1, y1, y2, y2) format for every image
    """

    filename = join(image_dir, filename)
    images = [join(image_dir, i) for i in os.listdir(
        image_dir) if is_image(i)]

    if len(images) != len(bboxes):
        print("WARNING: Images {} and bboxes {} not same length".format(len(images), len(bboxes)))

    with tf.io.TFRecordWriter(filename) as writer:

        for i, image_path in enumerate(images):
            bbox = bboxes[i]
            image = cv2.imread(image_path, 0)

            tf_example = bbox_to_tf_example(image, image_path, bbox)
            writer.write(tf_example.SerializeToString())

    print("Created tf-record file in: \n", filename)


def bbox_to_tf_example(image, filename, bboxes):

    # Parse image metas
    height, width, *_ = image.shape
    image_ext = ".png"

    encoded_image = cv2.imencode(image_ext, image)[1].tostring()

    # Parse bounding boxes
    xmins, xmaxs, ymins, ymaxs = [], [], [], []

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        xmins.append(x_min / width)
        xmaxs.append(x_max / width)
        ymins.append(y_min / height)
        ymaxs.append(y_max / height)

    classes_text = [b'Cell' for _ in bboxes]
    classes = [1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode("unicode_escape")),
        'image/source_id': bytes_feature(filename.encode("unicode_escape")),
        'image/encoded': bytes_feature(encoded_image),
        'image/format': bytes_feature(image_ext.encode("unicode_escape")),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def main(image_dir, annotations, annot_type, bbox_size=50):

    if annot_type.lower() == "xml":
        annot_df = extract_points_from_trackmate_xml(annotations, bbox_size)

        bboxes = [list(annot_df[annot_df.Frame == i].bbox)
                  for i in range(len(np.unique(annot_df.Frame)))]

    elif annot_type.lower() == "csv":
        annot_df = pd.read_csv(annotations)

        bboxes = [list(annot_df[annot_df.Frame == i].bbox)
                  for i in range(len(np.unique(annot_df.Frame)))]

    elif annot_type.lower() == "masks":
        pass
    else:
        print("Unsupported format {}".format(annot_type))

    write_tf_records(image_dir, bboxes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create TFrecords from given input format.')

    parser.add_argument('--image_dir', '-i', help="Path to directory with images.")
    parser.add_argument('--annotations', '-a', help="Path to file containing annotations")
    parser.add_argument('--format', '-f', help="Options: XML | CSV | MASKS")

    args = parser.parse_args()

    main(args.image_dir, args.annotations, args.format)
