export PYTHONPATH=.
export MODEL_S=resnet32
export MODEL_T=resnet110
#!/usr/bin/env bash

python ./pau_nn_representation/RepDistiller/train_student.py --path_t ./pau_nn_representation/RepDistiller/save/models/${MODEL_T}/ckpt_epoch_240.pth --distill crd --model_s $MODEL_S -a 0 -b 0.8 --trial 2
