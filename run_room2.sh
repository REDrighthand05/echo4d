#!/bin/bash
cd ~/workspace/echo_scene_app
P=".venv/bin/python -m echo_scene.pipeline"
J=jobs/room2
$P extract jobs/room2/input.mp4 $J && \
$P select $J && \
$P poses $J && \
$P markers $J && \
$P depth $J && \
$P fuse $J && \
$P mesh $J && \
$P present $J && echo ROOM2_ALL_DONE
