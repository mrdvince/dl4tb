import shutil
from pathlib import Path

import pandas as pd


def get_pandas_entry(id, path="data/train.csv"):
    df = pd.read_csv(path)
    classes = ["negative", "positive"]
    label = int(df[df["ID"] == id]["LABEL"])
    return classes[label], label


def copy_images_to_folder(data_dir):
    image_list = [path for path in Path(data_dir).rglob("*.png")]
    for idx, image in enumerate(image_list):
        image_name = image.name
        _, label = get_pandas_entry(image_name.split(".")[0])
        image_path = image.as_posix()
        if label == 0:
            # copy image to negative folder
            path = Path("data/proc_tb/negative")
            path.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, "data/proc_tb/negative" + "/" + image_name)
        if label == 1:
            # copy image to positive folder
            path = Path("data/proc_tb/positive")
            path.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, "data/proc_tb/positive" + "/" + image_name)


def verify_images_labels(name):
    return get_pandas_entry(name)


if __name__ == "__main__":
    copy_images_to_folder("data/tb_data/train")
