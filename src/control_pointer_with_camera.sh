#!/usr/bin/env bash


while [ "$1" != "" ]; do
    PARAMETERS="$PARAMETERS $1"
    shift
done
python3 computer_pointer_controller.py -i "autovideosrc ! videoconvert ! videoflip method=horizontal-flip ! appsink" $PARAMETERS
