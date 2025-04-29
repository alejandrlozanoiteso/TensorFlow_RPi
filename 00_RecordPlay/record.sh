#!/bin/bash

mkdir -p $1
cd $1

i=0
number=$2

while [ $i -lt $number ]
do
	echo "Speak now!!!!"
	arecord -Dplughw:3 -r16000 -c1 -fs16_le -d 2 "$1_$i".wav
	((i++))
	echo "$i"
done
