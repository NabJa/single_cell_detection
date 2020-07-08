"""
Script to train all models in training folder.
Training folder must contain a folder for each model named after the model.
Each model folder must contain a config file with naming convention: {model_name}.config

Example folder structure:

    models
        rcnn_brightfield
            rcnn_brightfield.config
        ssd_brightfield
            ssd_brightfield.config
"""

import argparse
from pathlib import Path
import subprocess
import time
from datetime import datetime


def main(train_path, log_name, ignore):
    models = [x for x in train_path.iterdir() if x.is_dir()]
    train_path = str(Path(__file__).parent.parent.joinpath("model_main.py"))

    if log_name.is_file():
        log_name.unlink()

    for model in models:
        if model.name in ignore:
            continue

        with open(log_name, 'a+') as log_file:
            log_file.write(f"Start training {model} {datetime.now()}\n")

        config_path = _find_config(model)

        tic = time.time()
        subprocess.run(["python", train_path,
                        f"--model_dir={model}", f"--pipeline_config_path={config_path}", "--alsologtostderr"])
        tac = time.time()

        with open(log_name, 'a') as log_file:
            log_file.write(f"FINISHED in {round(int(tac - tic) / 60)} minutes\n")


def _find_config(path, ignore="pipeline"):
    try:
        config_path = [x for x in path.glob("*.config") if not x.name.split(".")[0] in ignore]
    except StopIteration:
        raise StopIteration(f"No valid config file found in {path}")
    return config_path[0]


def _text_file(path):
    if path.endswith(".txt"):
        return Path(path)
    raise FileNotFoundError("Log file must be a text file ending with .txt!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", "-c", nargs="+", help="Path(s) to directory containing config "
                                                              "files. For more information type: python"
                                                              " train_all_models.py --docs")
    parser.add_argument("--log_path", "-l", type=_text_file, help="Optional: Path to save log file. Default=config_dir")
    parser.add_argument("-i", "--ignore", nargs="*", help="Optional: Folder to be ignored.")
    parser.add_argument("--docs", action="store_true", help="Show required config folder structure")
    args = parser.parse_args()

    if args.docs:
        print(__doc__)
        exit()

    for config in args.config_dir:
        CONFIG_DIR = Path(config)
        LOG_FILENAME = args.log_path if args.log_path else CONFIG_DIR.joinpath("train_log.txt")
        IGNORE = args.ignore if args.ignore else []

        main(CONFIG_DIR, Path(LOG_FILENAME), IGNORE)
