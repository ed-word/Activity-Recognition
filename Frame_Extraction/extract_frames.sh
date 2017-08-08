#!/bin/bash

EXPECTED_ARGS=1
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` video frames/sec [size=256]"
  exit $E_BADARGS
fi

FRAMES=$1


for folders in /home/edward/Documents/Windows/Eddy/Workspace/Hoopcam/Data/*
do

MAIN_NAME=`basename $folders`
echo $MAIN_NAME
mkdir -m 755 $MAIN_NAME

for subfolders in $folders/*
do

DIR_NAME=`basename $subfolders`
echo $DIR_NAME
mkdir -m 755 $MAIN_NAME/$DIR_NAME

for file in $subfolders/*
do
NAME=${file%.*}
BNAME=`basename $NAME`
echo $BNAME
mkdir -m 755 $MAIN_NAME/$DIR_NAME/$BNAME

ffmpeg -i $file -r $FRAMES $MAIN_NAME/$DIR_NAME/$BNAME/$BNAME.%4d.jpg
done

done

done
