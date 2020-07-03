#!/usr/bin/env bash


pip3 install --user pipenv
cd model
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001,face-detection-adas-binary-0001,gaze-estimation-adas-0002,head-pose-estimation-adas-0001,landmarks-regression-retail-0009
cd ../src
pipenv install
pipenv shell 
