#!/bin/bash

DIR=`pwd`

mkdir -p out/
rm -rf out/*
cd out/

cmake ../ -DCMAKE_INSTALL_PREFIX=$DIR

make -j8 install