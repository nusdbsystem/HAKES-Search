data_path="/data-ssd/guoyu/data"
program="./gen_base_params"
dataset=$1
output_dir=$2

case $dataset in
"gist-960")
  N=1000000
  NQ=1000
  d=960
  nlists=(1024 2048 4096 8192)
  drs=(1 2 4 8)
  ;;
"sphere-lmsys-768")
  N=1000000
  NQ=1000
  d=768
  nlists=(1024 2048 4096 8192)
  drs=(1 2 4 8)
  ;;
"dbpedia-openai")
  N=990000
  NQ=1000
  d=1536
  nlists=(1024 2048 4096 8192)
  drs=(1 2 4 8 16)
  ;;
"sphere-768-10m")
  N=9000000
  NQ=1000
  d=768
  nlists=(1024 2048 4096 8192)
  drs=(1 2 4 8)
  ;;
"sphere-1024-10m")
  N=9000000
  NQ=1000
  d=1024
  nlists=(1024 2048 4096 8192)
  drs=(1 2 4 8 16)
  ;;
"mobilenet-1024")
  N=1103593
  NQ=1000
  d=1024
  nlists=(1024 2048 4096 8192)
  drs=(1 2 4 8 16)
  ;;
"resnet-2048")
  N=1103593
  NQ=1000
  d=2048
  nlists=(1024 2048 4096 8192)
  drs=(1 2 4 8 16)
  ;;
*)
  echo "unknown dataset"
  exit 1
  ;;
esac

for nlist in ${nlists[@]}; do
  for dr in ${drs[@]}; do
    opq_out=$(expr $d / $dr)
    M=$(expr $opq_out / 2)
    echo "opq_out: $opq_out, M: $M"
    params_dir=${output_dir}/base_${dataset}-${dr}-${nlist}
    mkdir -p ${params_dir}
    ${program} ${N} ${d} ${data_path}/${dataset}/train.bin $opq_out $M $nlist ${params_dir} > ${params_dir}/log.out
  done
done
