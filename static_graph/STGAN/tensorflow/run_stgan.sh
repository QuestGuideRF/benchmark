#!bin/баш
if [ $
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|test 1|2(скорость|mem) /ssd3/benchmark_results/cwh/logs"
    exit
fi
function _set_params(){
    task="$1"
    index="$2"
    run_log_path=${3:-$(pwd)}
    model_name="STGAN"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    skip_steps=3
    keyword="Epoch:"
    separator=" "
    position=-2
    range=6
    model_mode=0
    mission_name="图像生成"
    direction_id=0
    run_mode=sp
    base_batch_size=32
    batch_size=${base_batch_size}
    log_file=${run_log_path}/log_${model_name}_${task}_${index}_${num_gpu_devices}
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    train_cmd=" --experiment_name 128 \
        --dataroot ./data/celeba \
        --n_d 1 \
        --batch_size ${batch_size} \
        --gpu ${gpu_id}"
    train_cmd="python -u поезд.py "${train_cmd}
    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 300
    kill -9 $train_pid
}
source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_run