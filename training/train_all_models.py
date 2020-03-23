"""
Script to train all models in training folder.
Training folder must contain a folder for each model named after the mode.
Each model folder must contain a config file with naming convention: {model_name}.config
"""

import argparse
import os
from os.path import join
from glob import glob
import subprocess
import time
from datetime import datetime

def main(train_path, log_name):

    MODELS = os.listdir(train_path)

    if os.path.isfile(LOG_FILENAME):
        os.remove(LOG_FILENAME)

    for model in MODELS:
        model_path = join(train_path, model)

        if not os.path.isdir(model_path):
            continue

        with open(log_name, 'a+') as log_file:
            log_file.write(f"Start training {model} {datetime.now()}\n")

        config_path = glob(join(model_path, "[!pipeline]*.config"))[0]

        tic = time.time()
        subprocess.run(["python", "train.py", "--logtostderr",
                        f"--train_dir={model_path}", f"--pipeline_config_path={config_path}"])
        tac = time.time()

        runtime = round(int(tac - tic)/60)

        with open(log_name, 'a') as log_file:
            log_file.write(f"FINISHED in {runtime} minutes\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", "-c", required=True)
    parser.add_argument("--log_path", "-l")

    args = parser.parse_args()

    CONFIG_DIR = args.config_dir
    LOG_FILENAME = args.log_path or "train_log.txt"

    main(CONFIG_DIR, LOG_FILENAME)
