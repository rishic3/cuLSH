#!/bin/bash

DIRS="culsh tests benchmark scripts"

if [[ "$1" == "--check" ]]; then
    black $DIRS --check
    isort $DIRS --profile black --check-only
    ruff check $DIRS
else
    black $DIRS
    isort $DIRS --profile black
    ruff check $DIRS --fix
fi
