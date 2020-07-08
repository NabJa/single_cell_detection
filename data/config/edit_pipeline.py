"""
Script to edit pipeline.config used in tf object detection api.
Parsing is based on protobuffer.
References:
    https://developers.google.com/protocol-buffers/docs/pythontutorial#writing-a-message
    https://www.tensorflow.org/api_docs/python/tf/io/gfile
"""

from pathlib import Path
import argparse
import sys
import tensorflow as tf

sys.path.append(r"C:\Users\N.Jabareen\Projects\TensorFlow\models\research\object_detection")
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


def read_config(path):
    config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile(str(path), "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    return config


def write_config(config, out):
    config_text = text_format.MessageToString(config)
    tf.io.write_file(str(out), config_text)


def set_config_train_paths(config, paths):
    config.train_input_reader.tf_record_input_reader.input_path[:] = paths


def set_config_eval_paths(config, paths):
    config.eval_input_reader[0].tf_record_input_reader.input_path[:] = paths


def main(config_path, train_records, val_records, output):
    train_records = Path(train_records)
    val_records = Path(val_records)
    output = Path(output)

    for train_records in train_records.glob("*.tfrecord"):

        config = read_config(config_path)

        # Extract meta, prepare folders
        microscope, cell_type, n_images = train_records.stem.split("_")

        train_path = output.joinpath(f"{microscope}_{cell_type}_{n_images}")
        train_path.mkdir()

        val_record = [str(val_records.joinpath(f"{microscope}.tfrecord"))]
        set_config_eval_paths(config, val_record)
        set_config_train_paths(config, [str(train_records)])

        write_config(config, train_path.joinpath(f"{train_path.name}.config"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_records", required=True)
    parser.add_argument("-v", "--validation_records", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-c", "--config", default="ssd.config", help="Default config to use. DEFAULT: ssd.config")

    args = parser.parse_args()
    main(args.config, args.train_records, args.validation_records, args.output)
