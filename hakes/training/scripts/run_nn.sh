dataset=$1
index_path=$2
device_no=$3
output_path=$4
lamb=$5
vt_lr=$6
pq_lr=$7
batch_size=$8

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
*)
    echo "unknown dataset"
    exit 1
    ;;
esac

echo "dataset: ${dataset}, N: ${N}, NQ: ${NQ}, d: ${d}"
echo "lamb: ${lamb}"

epochs=3

for nn in 60 50 40 30 20 10 5; do
    save_path=${output_path}/${dataset}-vt_lr-${vt_lr}-pq_lr-${pq_lr}-lamb-${lamb}-nn-${nn}
    mkdir -p ${save_path}
    python main.py --N ${N} --d ${d} --train_n 100000 --train_neighbor_count 100 --use_train_nn_count ${nn} --NQ ${NQ} --data_path data/${dataset} --index_path ${index_path} --epoch ${epochs} --batch_size ${batch_size} --vt_lr ${vt_lr} --pq_lr ${pq_lr} --lamb ${lamb} --device cuda:${device_no} --save_path_prefix ${save_path} >${save_path}/log.txt
    echo "nn: ${nn} done. saved at ${save_path}"
done
