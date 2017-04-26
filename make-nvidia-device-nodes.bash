#!/bin/bash

/sbin/modprobe nvidia

if [ "$?" -eq 0 ]; then
	NVDEVS=`lspci | grep -i NVDIA`
	N3D=`echo "$NVDEVS" | grep "3D controller" | wc -l`
	NVGA=`echo "$NVDEVS" | grep "VGA compatible controller" | wc -l`
	
	N=`expr $N3D + $NVGA`
	for i in `seq 0 $N`; do
		mknod -m 666 /dev/nvidia$i c 195 $i
	done
	
	mknod -m 666 /dev/nvidiactl c 195 255

else
	exit 1
fi

/sbin/modprobe nvidia-uvm

if [ "$?" -eq 0 ]; then
	D=`grep nvidia-uvm /proc/devices | awk '{print $1}'`

	mknod -m 666 /dev/nvidia-uvm c $D 0
else
	exit 1
fi
