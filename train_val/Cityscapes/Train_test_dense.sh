#!/bin/sh

## MODIFY PATH for YOUR SETTING
CAFFE_DIR=../../
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin
Train=1
Test=0

INITMODEL=./pspnet101_cityscapes.caffemodel
LOG_DIR=./log
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

if [ ${Train} -eq 1 ]; then 
  CMD="${CAFFE_BIN} train \
         --solver=./solver.prototxt \
         --gpu=0,1,2,3 \
         --snapshot=${INITMODEL}"
  echo Running ${CMD} && ${CMD}
fi



if [ ${Test} -eq 1 ]; then 
  TEST_ITER=`cat /home/jason/matlab/cvpr/mymodel/voc12/list/val.txt | wc -l`
  CMD="${CAFFE_BIN} test \
         --model=./ResNet_101_test_dilated_ext.prototxt \
         --gpu=0 \
         --weights=${MODEL} \
         --iterations=${TEST_ITER}"
  echo Running ${CMD} && ${CMD}
fi
