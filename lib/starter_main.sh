#!/bin/sh
cd /workdir

tmux new-session -d bash
tmux split-window -h bash
tmux split-window -t 0:0.0 -v bash

echo "Start Juputer: Start"
tmux send -t 0:0.0 "jupyter lab --allow-root --ip=0.0.0.0" C-m
echo "Start Juputer: Done"
