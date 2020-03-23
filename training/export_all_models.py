"""
Export all models in a given directory.
"""

import os
from os.path import join
import subprocess
import argparse
import re


def _dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def main(model_dir, out_dir):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for model in os.listdir(model_dir):

        max_model_number = 0
        config_file_name = None

        # Find latest model and config file
        for input_file in os.listdir(join(model_dir, model)):

            if ".config" in input_file and "pipeline" not in input_file:
                config_file_name = input_file
                continue

            if "model.ckpt-" in input_file:
                numbers = re.findall(r'\d+', input_file)
                if len(numbers) < 1:
                    continue

                if int(numbers[0]) > max_model_number:
                    max_model_number = int(numbers[0])

        if max_model_number == 0 or not config_file_name:
            print(f"\nWARNING: Could not find latest model in {model_dir}{model}")
            continue

        latest_model_path = join(model_dir, model, f"model.ckpt-{max_model_number}")
        config_path = join(model_dir, model, config_file_name)
        model_out_dir = join(out_dir, model)

        try:
            os.mkdir(model_out_dir)
        except:
            res = None
            while not res:
                res = input(f"\nOutput directory {model_out_dir} already exists. Do you want to override it? [y/n]")
                if res == "y":
                    os.rmdir(model_out_dir)
                    os.mkdir(model_out_dir)
                elif res == "n":
                    raise Exception()
                else:
                    print("Answer with \'y\' or \'n\' !")
                    res = None

        subprocess.run(["python", "export_inference_graph.py", "--input_type image_tensor",
                        "--pipeline_config_path", config_path,
                        "--trained_checkpoint_prefix", latest_model_path,
                        "--output_directory", model_out_dir],
                        check=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Export all model from a given directory.')

    parser.add_argument('--model_dir', type=_dir_path,
                        help="Path to directory with folders models.")
    parser.add_argument('--out_dir',
                        help="Path to directory to save all models.")

    args = parser.parse_args()

    main(args.model_dir, args.out_dir)
