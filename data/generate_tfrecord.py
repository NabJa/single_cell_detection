"""
Convert TrackMate XML data and corresponding images to TFRecord.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

import cv2
from . import trackmate_xml_to_csv as xml_to_csv
from . import tf_record_writer


def tf_record_example_generator(xml, images):

    points_and_bbox_df = xml_to_csv.extract_points_from_trackmate_xml(xml, [40])
    bbox_name = f"bbox{40}"

    image_paths = [x for x in images.glob("*.png")]
    frames = np.unique(points_and_bbox_df.Frame)

    for i, frame in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)}", end="\r")
        image_path = image_paths[frame]

        bboxes = list(points_and_bbox_df[points_and_bbox_df.Frame == frame][bbox_name])
        image = cv2.imread(str(image_path))

        tf_record_example = tf_record_writer.bbox_to_tf_example(image, str(image_path), bboxes)
        yield tf_record_example


def write_tf_record(path, record_generator):
    with tf.io.TFRecordWriter(path) as writer:
        for record in record_generator:
            writer.write(record.SerializeToString())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare tf record dataset based on TrackMate XML and images.")
    parser.add_argument("--xml", "-x", required=True, help="TrackMate XML file.")
    parser.add_argument("--images", "-i", required=True, help="Path to .png images.")
    args = parser.parse_args()

    XML_DIR = Path(args.xml)
    IMAGES = Path(args.images)

    RECORD_GENERATOR = tf_record_example_generator(XML_DIR, IMAGES)

    FILE_PATH = XML_DIR.parent.joinpath(f"{XML_DIR.stem}.tfrecord")
    write_tf_record(str(FILE_PATH), RECORD_GENERATOR)
    print("Saved tf_records in ", FILE_PATH)
