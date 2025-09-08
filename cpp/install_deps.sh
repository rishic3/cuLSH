#!/bin/bash

set -e

mkdir -p third_party
cd third_party

wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
mv eigen-3.4.0 eigen
rm eigen-3.4.0.tar.gz

echo "Eigen installed at third_party/eigen"
