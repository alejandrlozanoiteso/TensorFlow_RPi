#!/bin/bash

cd $1

i=0
number=$2

while [ $i -lt $number ]
do
	echo "playing $1_$i.wav"
	aplay "$1_$i".wav
	((i++))
done
