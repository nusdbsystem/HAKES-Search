data_path="/data-ssd/guoyu/data"
program="hakes_index"
dataset=$1
# specify the base path that contains the base parameters
base_path=$2
output_file=$3

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

echo "dataset: $dataset, N: $N, NQ: $NQ, d: $d"
echo "base_path: $base_path"
echo "output_file: $output_file"

dataset_path="${data_path}/${dataset}"
${program} ${N} ${NQ} ${d} 100 100 ${dataset_path}/train.bin ${dataset_path}/test.bin ${dataset_path}/neighbors.bin 1 2 8 0 1 100 50 ${base_path} > ${output_file}

