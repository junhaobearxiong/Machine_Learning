#!/bin/bash
python classify.py --mode train --algorithm adaboost --model-file easy.adaboost.model \
--data datasets/easy.train --num-boosting-iterations 10

python classify.py --mode test --model-file easy.adaboost.model --data datasets/easy.train \
--predictions-file easy.train.predictions

python compute_accuracy.py datasets/easy.train easy.train.predictions


python classify.py --mode train --algorithm adaboost --model-file bio.adaboost.model \
--data datasets/bio.train --num-boosting-iterations 10

python classify.py --mode test --model-file bio.adaboost.model --data datasets/bio.dev \
--predictions-file bio.dev.predictions

python compute_accuracy.py datasets/bio.dev bio.dev.predictions

python classify.py --mode train --algorithm adaboost --model-file vision.adaboost.model \
--data datasets/vision.train --num-boosting-iterations 10

python classify.py --mode test --model-file vision.adaboost.model --data datasets/vision.dev \
--predictions-file vision.dev.predictions

python compute_accuracy.py datasets/vision.dev vision.dev.predictions
