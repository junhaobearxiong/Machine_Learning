#!/bin/bash

python3 classify.py --mode train --algorithm logisticregression \
--model-file bio.logisticregression.model \
--data datasets/bio.train \
--online-learning-rate .01 \
--gd-iterations 20 \
--num-features-to-select 10

python3 classify.py --mode test --algorithm logisticregression \
--model-file bio.logisticregression.model \
--data datasets/bio.dev \
--predictions-file bio.dev.predictions

echo Accuracy for bio.dev
python3 compute_accuracy.py datasets/bio.dev bio.dev.predictions

python3 classify.py --mode train --algorithm logisticregression \
--model-file finance.logisticregression.model \
--data datasets/finance.train \
--online-learning-rate .01 \
--gd-iterations 20 \
--num-features-to-select 10

python3 classify.py --mode test --algorithm logisticregression \
--model-file finance.logisticregression.model \
--data datasets/finance.dev \
--predictions-file finance.dev.predictions

echo Accuracy for finance.dev
python3 compute_accuracy.py datasets/finance.dev finance.dev.predictions

python3 classify.py --mode train --algorithm logisticregression \
--model-file speech.logisticregression.model \
--data datasets/speech.train \
--online-learning-rate .01 \
--gd-iterations 20 \
--num-features-to-select 10

python3 classify.py --mode test --algorithm logisticregression \
--model-file speech.logisticregression.model \
--data datasets/speech.dev \
--predictions-file speech.dev.predictions

echo Accuracy for speech.dev
python3 compute_accuracy.py datasets/speech.dev speech.dev.predictions
