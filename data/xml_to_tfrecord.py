"""
Convert TrackMate XML data and corresponding images to TFRecord.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tensorflow as tf

import cv2
from data import trackmate_xml_to_csv as xml_to_csv
from data import tf_record_writer


def tf_record_example_generator(path, image_folder_name):
    directories = [x for x in path.iterdir() if x.is_dir()]

    for directory in directories:

        print(f"\tParse direcotry: {directory}")

        xml = next(directory.glob("*.xml"))
        image_dir = directory.joinpath(image_folder_name)

        points_and_bbox_df = xml_to_csv.extract_points_from_trackmate_xml(xml, [40])

        bbox_name = f"bbox{40}"

        image_paths = [x for x in image_dir.glob("*.png")]
        frames = np.unique(points_and_bbox_df.Frame)

        for frame in tqdm(frames):
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

    parser = argparse.ArgumentParser(description="Prepare tfdata")

    parser.add_argument("--data_dir", "-d", required=True)
    parser.add_argument("--image_folder_name", "-i", required=True)
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    IMAGE_NAME = args.image_folder_name

    FILENAME = f"{IMAGE_NAME}.tfrecord"
    FILE_PATH = DATA_DIR.joinpath(FILENAME)

    RECORD_GENERATOR = tf_record_example_generator(DATA_DIR, IMAGE_NAME)
    write_tf_record(str(FILE_PATH), RECORD_GENERATOR)
