#!/bin/sh

## MODIFY PATH for YOUR SETTING
CAFFE_DIR=../../
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin
Train=1
Test=0

INITMODEL=./init.caffemodel
LOG_DIR=./log
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

if [ ${Train} -eq 1 ]; then 
  CMD="${CAFFE_BIN} train \
         --solver=./solver_dcu.prototxt \
         --gpu=0 \
         --weights=${INITMODEL}"
  echo Running ${CMD} && ${CMD}
fi



if [ ${Test} -eq 1 ]; then 
  TEST_ITER=`cat ./val.txt | wc -l`
  CMD="${CAFFE_BIN} test \
         --model=./test.prototxt \
         --gpu=0 \
         --weights=${MODEL} \
         --iterations=${TEST_ITER}"
  echo Running ${CMD} && ${CMD}
fi
