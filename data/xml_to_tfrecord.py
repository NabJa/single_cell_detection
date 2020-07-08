"""
Convert TrackMate XML data and corresponding images to TFRecord.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

import cv2
from data import trackmate_xml_to_csv as xml_to_csv
from data import tf_record_writer


def tf_record_example_generator(annot_path, image_folder_name, cell_ignore):

    cell_directories = [cell for cell in annot_path.iterdir() if cell.is_dir() and (cell.name not in cell_ignore)]

    for cell_dir in cell_directories:
        for experiment in cell_dir.iterdir():

            print(f"\tParse direcotry: {experiment}")

            xml = [x for x in experiment.glob("*.xml")]
            if len(xml) < 1:
                print(f"\t\tWARNING: No xml found in {experiment}. Skipping experiment...")
                break
            xml = xml[0]

            image_dir = experiment.joinpath(image_folder_name)

            if experiment.name == "20200203_3T3_Hoechst_Pos7":
                a = 5

            points_and_bbox_df = xml_to_csv.extract_points_from_trackmate_xml(xml, [40])

            bbox_name = f"bbox{40}"

            image_paths = [x for x in image_dir.glob("*.png")]
            frames = np.unique(points_and_bbox_df.Frame)

            for frame in frames:
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

    parser.add_argument("--annotation_dir", "-a", required=True, help="Path to annotation. Containg subfolders -> "
                                                                      "cell_type -> experiment_name")
    parser.add_argument("--target_folder", "-t", required=True, help="Directory name containing images")
    parser.add_argument("--ignore_cell", "-ic", nargs="*", help="Cell types to ignore")
    args = parser.parse_args()

    DATA_DIR = Path(args.annotation_dir)
    RECORD_GENERATOR = tf_record_example_generator(DATA_DIR, args.target_folder, args.ignore_cell)

    FILENAME = f"{args.target_folder}.tfrecord"
    FILE_PATH = DATA_DIR.joinpath(FILENAME)
    write_tf_record(str(FILE_PATH), RECORD_GENERATOR)
