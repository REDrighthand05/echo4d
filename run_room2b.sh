#!/bin/bash
cd ~/workspace/echo_scene_app
P=".venv/bin/python -m echo_scene.pipeline"
J=jobs/room2
$P poses $J && \
$P markers $J && \
$P depth $J && \
$P fuse $J && \
$P mesh $J && \
$P present $J && echo ROOM2B_ALL_DONE
