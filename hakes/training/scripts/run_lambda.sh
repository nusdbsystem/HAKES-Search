dataset=$1
index_path=$2
device_no=$3
output_path=$4
batch_size=$5

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

epochs=5

# nn=100
nn=50

for lr in 1e-3 1e-4 1e-5; do
    for lamb in -1 0 0.01 0.03 0.1 0.3 1 3 10 30; do
        save_path=${output_path}/${dataset}-lr-${lr}-lamb-${lamb}
        mkdir -p ${save_path}
        python main.py --N ${N} --d ${d} --train_n 100000 --train_neighbor_count 100 --use_train_nn_count ${nn} --NQ ${NQ} --data_path data/${dataset} --index_path ${index_path} --epoch ${epochs} --batch_size ${batch_size} --vt_lr ${lr} --pq_lr ${lr} --lamb ${lamb} --device cuda:${device_no} --save_path_prefix ${save_path} >${save_path}/log.txt
        echo "lamb: ${lamb} done. saved at ${save_path}"
    done
done
