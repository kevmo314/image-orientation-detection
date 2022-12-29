#!/bin/bash

mkdir train
mkdir test

cat train.txt | xargs -P $(nproc) -I{} ./yt-dlp -f 243 -o 'yt-%(id)s.webm' -N $(nproc) {}
ls *.webm | xargs -P $(nproc) -I{} ffmpeg -i {} -vf scale=160:90:flags=fast_bilinear -r 0.25 train/{}_%04d.png
rm *.webm

cat test.txt | xargs -P $(nproc) -I{} ./yt-dlp -f 243 -o 'yt-%(id)s.webm' -N $(nproc) {}
ls *.webm | xargs -P $(nproc) -I{} ffmpeg -i {} -vf scale=160:90:flags=fast_bilinear -r 0.25 test/{}_%04d.png
rm *.webm
