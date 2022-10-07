#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

wget wget https://dl.dropboxusercontent.com/s/1p03h44a1w9c2re/models.tar.xz -P $SCRIPTPATH/sd-v1-4-onnx/
echo "Extracting models to {$SCRIPTPATH}/sd-v1-4-onnx/models/"
cd $SCRIPTPATH/sd-v1-4-onnx/ && tar -xvf models.tar.xz
rm $SCRIPTPATH/sd-v1-4-onnx/models.tar.xz
