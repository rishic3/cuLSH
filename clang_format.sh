#!/bin/bash

find cpp -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i -style=file:.clang-format
