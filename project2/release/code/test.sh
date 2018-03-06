#!/bin/bash

python3 classify.py --mode train --algorithm logisticregression \
--model-file bio.logisticregression.model \
--data datasets/bio.train \
--online-learning-rate .05 \
--gd-iterations 50 \
--num-features-to-select 20

python3 classify.py --mode test --algorithm logisticregression \
--model-file bio.logisticregression.model \
--data datasets/bio.dev \
--predictions-file bio.dev.predictions

python3 compute_accuracy.py datasets/bio.dev bio.dev.predictions
