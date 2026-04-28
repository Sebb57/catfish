#!/bin/bash
curl -L -o ./chess-evaluations.zip\
  https://www.kaggle.com/api/v1/datasets/download/ronakbadhe/chess-evaluations
pid=$!
wait $(pid)
unzip chess-evaluations.zip
