#!/usr/bin/env bash


while [ "$1" != "" ]; do
    PARAMETERS="$PARAMETERS $1"
    shift
done
python3 computer_pointer_controller.py -i ../bin/vlcsnap-2020-05-09-13h30m32s960.png $PARAMETERS
