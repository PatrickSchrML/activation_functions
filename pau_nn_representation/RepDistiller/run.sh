#!/usr/bin/env bash

export PYTHONPATH=.
export MODEL_S=vgg8_pau
export MODEL_T=vgg13_vanilla

python ./pau_nn_representation/RepDistiller/train_student.py --path_t ./pau_nn_representation/RepDistiller/save/models/${MODEL_T}/ckpt_epoch_240.pth --distill crd --model_s $MODEL_S -a 0 -b 0.8 --trial 2