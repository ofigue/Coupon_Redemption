export TRAINING_DATA=input/trainsetBalanced.csv # BEFORE FOLDS
# export TEST_DATA=input/test.csv

export MODEL=$1

python -m src.gridSearch
