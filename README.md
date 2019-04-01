#Dense Convolutional Networks for Semantic Segmentation

by Chaoyi Han, Yiping Duan, Xiaoming Tao, and Jianhua Lu.


##Introduction
This repository is modified from '[Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)' and [DeepLab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2). The implementation mainly involves the "DCU" layer and corresponding group softmax normalizations. Currently, the group normalization explicitly uses a combination of softmax layers in official caffe.

##Usage 
For quick start, we provide training and validation scripts on Cityscapes dataset. The example is the Res101-SPP-DCU structure introduced in our paper.  Training scripts on other sturctures as well as datasets should be strightforward with minor modifications. 

##Installation
    - Clone this repo:

   ```
   git clone https://github.com/hancy16/DCU
   cd DCU
   ```

    - Follow the instructions of [Caffe](https://github.com/BVLC/caffe) and [DeepLab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) for installation. 
The code is tested on Ubuntu 16.04 with CUDA 8.0.

##Training
1. Get in the subdirectory:

   ```shell
   cd trainval/Cityscapes/
   ```

2. Preparing the training Scripts on Cityscapes:
   - Modify the variable 'path' in init.sh, this contains your path to the Cityscapes dataset. The script will automatically generate the prototxt file, 'pspnet101_cityscapes_473_dcu.prototxt' and the data list, 'list/train.txt'. For any problem concerning the data list please refer to 'list/trainval_list_examples.sh'.


   - Run the scripts:
   ```
   ./init.sh 
   ```

   - Download the initmodel for pretraining, e.g. 'pspnet101_cityscapes.caffemodel' for Res101-SPP-DCU. Replace the variable 'INITMODEL' in 'train_test.sh' with the path to your initmodel.

3. Run the training scripts:

   ```
   sh ./train_test.sh
   ```
##Testing
 We adopt the evaluation code in [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) with some modifications. Multiscale testing and mirror testing are included in our evaluations.  For how to use the scripts please refer to the PSPNET.README. 


## Contact
 For any problems please contact hancy16@mails.tsinghua.edu.cn.
