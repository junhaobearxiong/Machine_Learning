#!/bin/bash

python classify.py --mode train --algorithm adaboost --model-file speech.adaboost.model \
--data datasets/speech.train --num-boosting-iterations 10

