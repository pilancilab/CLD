# CLD
Convex Language Detection for Low Resource Languages

## Introduction

This repo contains a distilled version of using Convex NN (cvxNN) for binary classification of low resource languages in JAX. It serves as an entry point to first classify between 2 dialects, then scale up to 5 after building familiarity. 

1) `cronos_trainer.py` -> entry point for training cvxNN models
2) `defrun.py` -> handles actual work, dataloading and input directories handled in dataloader utils
3) `solve` -> contains the convex reformulation method and solvers in JAX
4) `asr_pipeline` -> demos training a vanilla NN detection head layer on top of Whisper, and then attaching it to override Whipser's retrival head

## Goal 

Let the input to `cronos_trainer.py` be 2 dialects, such as `en-US` and `es-SIN`. Assume these are labeled as 0 and 1, respectively. We first train the cvxNN model to binary classify between these two dialects robustly, then save the model as a tiny pickle file (output of `cronos_trainer.py`).

We will show that by using the tiny saved cvxNN model in the `asr_pipeline\asr_test2.py` pipeline, we can achieve better performance than the current vanilla NN model.

## Metrics

We will define 'better performance' as: 
  - higher classification accuracy
  - lower TFOPS, lower mem requirements
  - faster speed at inference
  - also faster training time (no hyperparameter grid search needed)
  - etc etc

## To Do 

 - Let's onboard with JAX, cvxNN, and binary classification of 2 dialects in this repo in a clean way
 - Add 3 more dialects to each langauge
 - Make plots for metrics and upload into Overleaf (https://www.overleaf.com/4838225339djtytwywdtmd#556cb6), think of a funny name, and we're done!

 ## Other Resources
 
  - `https://github.com/pilancilab/CRONOS` contains examples of multi-class classification
  - if `jaxtest.py` works, then JAX is installed correctly 
  - must use NVIDIA GPUs
  - for this starting point, can use any two inputs (ie Singlish and English, or Beijing Mandarin and Standard Mandarin, etc), as long as the dataset is balanced (don't worry about size of dataset, we'll scale up later)
  - size of individual data samples is more important (as large as possible, but still fitting on VRAM)
  - the real advantage of this solver class is it's **fast**, since there is no need for huge matvecs (unlike vanilla NN)

