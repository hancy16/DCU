#!/bin/usr/bash

#Generate the train_val list 

CUR_DIR=`pwd`
DATA_PATH=/path/to/dataset/
cd $DATA_PATH
FILENUM =0
for f in ./cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/*/*.png; do
    echo $f >> OriImage.txt
    ((FILENUM++))
done


for j in ./cityscapes/gtFine_trainvaltest/gtFine/train/*/*labelTrainIds.png; do   
   echo $j >> SegImage.txt
done

for((i=1;i<=$FILENUM;i++))
do
    Img=$(sed -n "$i p" OriImage.txt)
    Seg=$(sed -n "$i p" SegImage.txt)
    echo "$Img $Seg" >>train.txt
    printf "Processing: $i\n"
done

mv  $DATA_PATH/train.txt $CUR_DIR
rm OriImage.txt
rm SegImage.txt


