# kevmo314/image-orientation-detection

This is a simple image orientation detection model. It uses a CNN to identify upside down images.

It is trained on YouTube videos. This repository is intended to be educational as the model is not particularly complex or innovative, but instead minimally implemented. See more details in the corresponding [blog post](https://blog.kevmo314.com/image-orientation-detection.html).

## Dataset

Run `./download.sh` to download the dataset. [yt-dlp](https://github.com/yt-dlp/yt-dlp) is required to download the videos.

## Training

Run `python run.py` to train the model. It will output the model to `model.pt` in PyTorch format.