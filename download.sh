#!/bin/bash

mkdir train
mkdir test

cat train.txt | xargs -P $(nproc) -I{} python3 download.py {} train
cat test.txt | xargs -P $(nproc) -I{} python3 download.py {} test
