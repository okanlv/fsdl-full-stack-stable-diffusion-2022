#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

wget https://huggingface.co/kamalkraj/stable-diffusion-v1-4-onnx/resolve/main/models.tar.gz -P $SCRIPTPATH/sd-v1-4-onnx/

echo "Extracting models to {$SCRIPTPATH}/sd-v1-4-onnx/models/"
cd $SCRIPTPATH/sd-v1-4-onnx/ && tar -xvzf models.tar.gz
rm $SCRIPTPATH/sd-v1-4-onnx/models.tar.gz
