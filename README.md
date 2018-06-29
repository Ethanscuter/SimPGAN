Note: this code haven't been rewrite so that everyone one can use freely. You can use this code, but you should change the path of model files and data files in your own environment before running the code. However, feel free to let me konw if you have any problem.

This repository contains code for the KSEM 2018 paper [
Cross-dataset Person Re-Identification Using Similarity Preserved Generative Adversarial Networks](https://arxiv.org/abs/1806.04533).

## About code
1. This code was writen based on PyTorch framework 0.3.
2. Before you run this code, you need download the dataset first. Howerver, I have already upload train.list in the data directory. This is the file you need in the preprocessing.

## How to run

python main.py

Note: you can change the hyperparameters in the main.py.

## How to visualize

There is a file name show.html in the sample directory. Open the show.html in your browser after you run python main.py. It will auto plot the intermediate images while training.




