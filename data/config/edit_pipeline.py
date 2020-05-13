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


def main(config, records):
    records = Path(records)

    for eval_record in records.iterdir():

        record_parts = eval_record.name.split("_")
        if "ignore" in record_parts:
            continue

        microscopy, cell_type = record_parts
        cell_type = cell_type.split(".")[0]
        record_parts.insert(1, "ignore")
        train_name = "_".join(record_parts)

        train_path = eval_record.parent.joinpath(train_name)

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(str(config), "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [str(eval_record)]
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [str(train_path)]

        # Write new config
        config_text = text_format.MessageToString(pipeline_config)
        tf.io.write_file(str(records.joinpath(f"ssd_{microscopy}_{cell_type}.config")), config_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Default config to use. (e.g. ssd.config)")
    parser.add_argument("-p", "--path")

    args = parser.parse_args()
    main(args.config, args.path)
