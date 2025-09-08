#!/bin/bash

set -e

DATA_DIR="data/sift"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$DATA_DIR"

SIFT_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
TARBALL_PATH="$DATA_DIR/sift.tar.gz"

wget -O "$TARBALL_PATH" "$SIFT_URL"
tar -xzf "$TARBALL_PATH" -C "$DATA_DIR" --strip-components=1
rm "$TARBALL_PATH"

echo "SIFT1M downloaded to: $DATA_DIR"
