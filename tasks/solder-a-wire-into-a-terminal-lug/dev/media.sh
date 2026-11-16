#!/bin/bash
# Dev only: render a logged run's three cameras and collect them with the first top-camera frame. dev/media.sh <label>
source ~/IsaacLab-Arena/yam.env
D=~/yam-eval/lug-solder-"$1"
inspect-robots video "$D"/adhoc_*.json > /dev/null
mkdir -p "$D"-media
cp "$D"/frames/*/*.mp4 "$D"/frames/*/*top_cam_000000.npy "$D"-media/
ls "$D"-media
