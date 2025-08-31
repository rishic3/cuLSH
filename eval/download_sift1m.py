import argparse
import os
import shutil
import tarfile
import urllib.request as request
from contextlib import closing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/sift")
    args = parser.parse_args()

    data_dir = args.dir
    os.makedirs(data_dir, exist_ok=True)

    with closing(
        request.urlopen("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz")
    ) as r:
        with open(os.path.join(data_dir, "sift.tar.gz"), "wb") as f:
            shutil.copyfileobj(r, f)

    tar = tarfile.open(os.path.join(data_dir, "sift.tar.gz"), "r:gz")
    tar.extractall()
