"""
Script to train all models in training folder.
Training folder must contain a folder for each model named after the mode.
Each model folder must contain a config file with naming convention: {model_name}.config
"""

import argparse
from pathlib import Path
import subprocess
import time
from datetime import datetime


def main(train_path, log_name):

    models = [x for x in train_path.iterdir() if x.is_dir()]
    train_path = str(Path(__file__).parent.joinpath("train.py"))

    if log_name.is_file():
        log_name.unlink()

    for model in models:
        with open(log_name, 'a+') as log_file:
            log_file.write(f"Start training {model} {datetime.now()}\n")

        config_path = next(model.glob("[!pipeline]*.config"))

        tic = time.time()
        subprocess.run(["python", train_path, "--logtostderr",
                        f"--train_dir={model}", f"--pipeline_config_path={config_path}"])
        tac = time.time()

        with open(log_name, 'a') as log_file:
            log_file.write(f"FINISHED in {round(int(tac - tic)/60)} minutes\n")


def _text_file(path):
    if path.endswith(".txt"):
        return Path(path)
    raise FileNotFoundError("Log file must be a text file ending with .txt!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", "-c", required=True)
    parser.add_argument("--log_path", "-l", type=_text_file)

    args = parser.parse_args()

    CONFIG_DIR = Path(args.config_dir)
    LOG_FILENAME = args.log_path if args.log_path else CONFIG_DIR.joinpath("train_log.txt")

    main(CONFIG_DIR, Path(LOG_FILENAME))
