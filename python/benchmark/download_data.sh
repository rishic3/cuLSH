#!/bin/bash

set -e

usage() {
    echo "Usage: $0 <dataset> [--dir <data_dir>]"
    echo ""
    echo "Datasets:"
    echo "  sift     - SIFT1M: 1M 128-dim vectors for RPLSH"
    echo "  kosarak  - Kosarak: ~990K transactions for MinHash"
    echo ""
    echo "Options:"
    echo "  --dir <path>  - Directory to save dataset (default: data/<dataset>)"
    exit 1
}

download_sift() {
    local DATA_DIR="$1"
    local SIFT_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    local TARBALL_PATH="$DATA_DIR/sift.tar.gz"

    wget -O "$TARBALL_PATH" "$SIFT_URL"
    tar -xzf "$TARBALL_PATH" -C "$DATA_DIR" --strip-components=1
    rm "$TARBALL_PATH"

    echo "SIFT1M downloaded to: $DATA_DIR"
}

download_kosarak() {
    local DATA_DIR="$1"
    local KOSARAK_URL="https://fimi.uantwerpen.be/data/kosarak.dat.gz"
    local GZ_PATH="$DATA_DIR/kosarak.dat.gz"

    wget -O "$GZ_PATH" "$KOSARAK_URL"
    gunzip -f "$GZ_PATH"

    echo "Kosarak downloaded to: $DATA_DIR"
}

if [[ $# -lt 1 ]]; then
    usage
fi

DATASET="$1"
shift

DATA_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    DATA_DIR="data/$DATASET"
fi

mkdir -p "$DATA_DIR"

case "$DATASET" in
    sift)
        download_sift "$DATA_DIR"
        ;;
    kosarak)
        download_kosarak "$DATA_DIR"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo ""
        usage
        ;;
esac

