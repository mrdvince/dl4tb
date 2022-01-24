from fileinput import filename
from pathlib import Path
import gdown

import zipfile


def get_data(url, path, filename):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(path / filename), quiet=False)
    # extract the zip file
    unzip(path, filename)


def unzip(path, filename):
    with zipfile.ZipFile(str(path / filename), "r") as zip_ref:
        zip_ref.extractall(path)


if __name__ == "__main__":
    get_data(
        "https://drive.google.com/uc?id=1KdpV3M27kV-_QOQOrAentfzZ2tew8YS-&",
        "data",
        "tb_data.zip",
    )
