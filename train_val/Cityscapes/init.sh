#!/bin/sh
## A SIMPLE SHELL FILE FOR PREPARING YOUR TRAINING SCRIPTS 
## MODIFY PATH for YOUR SETTING
path='/path/to/dataset/'
cp ./prototxt/trainval_example.prototxt pspnet101_cityscapes_473_dcu.prototxt
cp ./list/trainval_list_example.sh ./list/trainval_list.sh
sed -i "s#\/path\/to\/dataset\/#$path#" pspnet101_cityscapes_473_dcu.prototxt
sed -i "s#\/path\/to\/dataset\/#$path#" ./list/trainval_list.sh
sh ./list/trainval_list.sh
