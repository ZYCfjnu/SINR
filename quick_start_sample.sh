#!/bin/bash
python generate.py

rm -rf ./experiments/customDataset/evaluation/*ori*

python Matcher.py

rm -rf ./experiments/customDataset/evaluation/*