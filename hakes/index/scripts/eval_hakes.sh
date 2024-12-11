#!/bin/bash

# Set the folder path
data_path="/data-ssd/guoyu/data"
program="hakes_index"
dataset=$1
output_dir=$2
# specify the base path that contains the base parameters
base_path=$3
# specify the update path that contains the trained parameters
update_path=$4

topk=10

case $dataset in
"gist-960")
    N=1000000
    NQ=1000
    d=960
    ;;
"sphere-lmsys-768")
    N=1000000
    NQ=1000
    d=768
    ;;
"dbpedia-openai")
    N=990000
    NQ=1000
    d=1536
    ;;
"sphere-768-10m")
    N=9000000
    NQ=1000
    d=768
    ;;
"sphere-1024-10m")
    N=9000000
    NQ=1000
    d=1024
    ;;
"mobilenet-1024")
    N=1103593
    NQ=1000
    d=1024
    ;;
"resnet-2048")
    N=1103593
    NQ=1000
    d=2048
    ;;
*)
    echo "unknown dataset"
    exit 1
    ;;
esac

dataset_path=${data_path}/${dataset}

echo "${program} ${N} ${NQ} ${d} ${topk} 100 ${dataset_path}/train.bin ${dataset_path}/test.bin ${dataset_path}/neighbors.bin 1 1 8 0 1 100 50 ${base_path} ${update_path} > ${output_dir}/${dataset}-eval.out"
${program} ${N} ${NQ} ${d} ${topk} 100 ${dataset_path}/train.bin ${dataset_path}/test.bin ${dataset_path}/neighbors.bin 1 1 8 0 1 100 50 ${base_path} ${update_path} >${output_dir}/${dataset}-eval.out
echo "${dataset} done"
