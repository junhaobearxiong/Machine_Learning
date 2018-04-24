#!/bin/bash
python classify.py --mode train --algorithm lambda_means --model-file easy.adaboost.model \
--data ../datasets/easy.train

python classify.py --mode test --model-file easy.adaboost.model --data ../datasets/easy.train \
--predictions-file easy.train.predictions

#python compute_accuracy.py datasets/easy.train easy.train.predictions


