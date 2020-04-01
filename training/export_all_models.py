"""
Export all models in a given directory.
"""

from pathlib import Path
import subprocess
import argparse


def _dir_path(string):
    if Path(string).is_dir():
        return string
    raise NotADirectoryError(string)


def find_latest_model(path):
    path = Path(path)
    checkpoints = [str(x).split("ckpt-")[1].split(".")[0] for x in path.glob(r"*ckpt*")]
    checkpoints = [int(x) for x in checkpoints]
    return max(checkpoints)


def main(model_dir, out_dir):

    out_dir.mkdir(exist_ok=True)
    model_subdirs = [x for x in model_dir.iterdir() if x.is_dir()]

    for model in model_subdirs:

        max_model_number = find_latest_model(model)
        config_path = next(model.glob("[!pipeline]*.config"))

        model = model.joinpath(f"model.ckpt-{max_model_number}")
        model_out_dir = out_dir.joinpath(model.name)

        subprocess.run(["python", "export_inference_graph.py", "--input_type image_tensor",
                        "--pipeline_config_path", config_path,
                        "--trained_checkpoint_prefix", model,
                        "--output_directory", model_out_dir], check=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Export all models from a given directory.')

    parser.add_argument("--model_dir", "-m", type=_dir_path,
                        help="Path to directory with folders models.")
    parser.add_argument("--out_dir", "-o",
                        help="Path to directory to save all models.")

    args = parser.parse_args()

    main(Path(args.model_dir), Path(args.out_dir))
