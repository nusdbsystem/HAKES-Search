data_path="/data-ssd/guoyu/data"
dataset=$1
output_dir=$2
topk=10

case $dataset in
"gist-960")
  N=1000000
  NQ=1000
  d=960
  M=120
  nlist=1024
  nprobe=200
  k_factor=100 
  base_path="base-params-path"
  update_path="trained-params-path"
  ;;
"sphere-lmsys-768")
  N=1000000
  NQ=10000
  d=768
  M=96
  nlist=1024
  nprobe=100 
  k_factor=100
  base_path="base-params-path"
  update_path="trained-params-path"
  ;;
"dbpedia-openai")
  N=990000
  NQ=10000
  d=1536
  M=96
  nlist=1024
  nprobe=200 
  k_factor=50
  base_path="base-params-path"
  update_path="trained-params-path"
  ;;
"sphere-768-10m")
  N=9000000
  NQ=1000
  d=768
  M=96
  nlist=8192
  nprobe=300 
  k_factor=100
  base_path="base-params-path"
  update_path="trained-params-path"
  ;;
"sphere-1024-10m")
  N=9000000
  NQ=1000
  d=1024
  M=128
  nlist=8192
  nprobe=600 
  k_factor=300
  base_path="base-params-path"
  update_path="trained-params-path"
  ;;
"mobilenet-1024")
  N=1103593
  NQ=20000
  d=1024
  M=128
  nlist=2048
  nprobe=100 
  k_factor=20
  base_path="base-params-path"
  update_path="trained-params-path"
  ;;
"resnet-2048")
  N=1103593
  NQ=1000
  d=2048
  M=128
  nlist=1024
  nprobe=50 
  k_factor=20
  base_path="base-params-path"
  update_path="trained-params-path"
  ;;
*)
  echo "unknown dataset"
  exit 1
  ;;
esac

# nclient=1
nclient=32
for write_ratio in 0 0.1 0.2 0.3 0.4 0.5; do
  ${program} ${N} ${NQ} ${d} ${topk} 100 ${dataset_path}/train.bin ${dataset_path}/test.bin ${dataset_path}/neighbors.bin 1 0 ${nclient} ${write_ratio} 1 ${nprobe} ${k_factor} ${base_path} ${update_path} >${output_dir}/rw/lake_trained-${dataset}-nc${nclient}-write${write_ratio}.out
  echo "write_ratio: ${write_ratio} done"
done
echo "${dataset} done"
