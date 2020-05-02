import argparse
from glob import glob
from os.path import join, basename, dirname

import numpy as np

from tqdm import tqdm

import tensorflow as tf

import cv2
import trackmate_xml_to_csv as xml_to_csv
import image_slice_utils as slicer
import tf_record_writer


def tf_record_example_generator(path, image_folder_name, number_of_tiles, bbox_size):
    directories = glob(join(path, "*\\"))
    assert number_of_tiles % 2 == 0, "Error in splitting images. Uneven number of images requested."

    for directory in directories:

        print(f"\tParse direcotry: {directory}")

        xmls = glob(join(directory, "*.xml"))
        image_dir = glob(join(directory, image_folder_name))

        assert len(xmls) == 1, f"Folder {basename(dirname(directory))} should contain only one xml file."
        assert len(image_dir) == 1, f"Folder {basename(dirname(directory))} should contain only one image folder {image_folder_name}."

        points_and_bbox_df = xml_to_csv.extract_points_from_trackmate_xml(xmls[0], [bbox_size])

        bbox_name = f"bbox{bbox_size}"

        image_paths = glob(join(image_dir[0], "*.png"))
        frames = np.unique(points_and_bbox_df.Frame)
        for frame in tqdm(frames):
            image_path = image_paths[frame]

            bboxes = list(points_and_bbox_df[points_and_bbox_df.Frame == frame][bbox_name])
            image = cv2.imread(image_path)

            tiles = slicer.split_image_with_bboxes_efficient(bboxes, image, bbox_size, number_of_tiles)

            for tile_key, annotation in tiles.items():

                tile_image = annotation["image"]
                tile_bboxes = annotation["bboxes"]
                tile_name = "".join(map(str, tile_key))

                tf_record_example = tf_record_writer.bbox_to_tf_example(tile_image, f"{tile_name}_{image_path}", tile_bboxes)
                yield tf_record_example


def write_tf_record(path, record_generator):
    with tf.io.TFRecordWriter(path) as writer:
        for record in record_generator:
            writer.write(record.SerializeToString())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare tfdata")

    parser.add_argument("--data_dir", "-d", required=True)
    parser.add_argument("--image_folder_name", "-i", required=True)
    parser.add_argument("--tiles", "-t", type=int, default=[0], nargs="*")
    parser.add_argument("--bboxes", "-b", type=int, default=[20], nargs="*")

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    IMAGE_NAME = args.image_folder_name

    for tile in args.tiles:
        for bbox in args.bboxes:

            print(f"Generating tfrecord Tiles:{tile} BBox:{bbox}")

            FILENAME = f"{IMAGE_NAME}_tiles{tile}_bboxes{bbox}.tfrecord"
            FILE_PATH = join(DATA_DIR, FILENAME)

            RECORD_GENERATOR = tf_record_example_generator(DATA_DIR, IMAGE_NAME, tile, bbox)
            write_tf_record(FILE_PATH, RECORD_GENERATOR)
