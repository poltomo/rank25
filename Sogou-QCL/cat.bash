#!/bin/bash

mkdir -p ./temp
mkdir -p ./main_v2
cd main_v2

for i in {2..9}
do 
    echo start to process ${i} shard
    cat ../main_v2_slices/SogouQCL.${i}.v2.bz2.tar.gz.* > ../temp/${i}.tar.gz
    echo complete cat SogouQCL.${i}.v2.bz2.tar.gz
    tar -zxf ../temp/${i}.tar.gz
    echo complete untar SogouQCL.${i}.v2.bz2
done

rm -rf ../temp

echo complete all