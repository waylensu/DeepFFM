#!/bin/bash

tmux kill-session -t deepffm
ps -ef|grep worker.py|awk '{print $2}' | xargs kill -9
