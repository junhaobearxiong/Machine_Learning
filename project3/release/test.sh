#!/bin/bash

python classify.py --mode train --algorithm adaboost --model-file easy.adaboost.model \
--data datasets/easy.train --num-boosting-iterations 10

python classify.py --mode test --model-file easy.adaboost.model --data datasets/easy.dev \
--predictions-file easy.dev.predictions
