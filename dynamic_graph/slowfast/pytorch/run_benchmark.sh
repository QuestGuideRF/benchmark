#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 sp|mp"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=8
    model_name="SlowFast"
    run_mode=${2:-"sp"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    mission_name="视频分类"
    direction_id=0
    skip_steps=10
    keyword="batch_cost:"
    separator=" "
    position=5
    model_mode=0
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}
    batch_size=`expr $base_batch_size \* $num_gpu_devices`
}
function _train(){
     train_cmd="--cfg МЕДЛЕННЫЙ.yaml
                DATA.PATH_TO_DATA_DIR данные/все
                TRAIN.BATCH_SIZE ${batch_size}
                NUM_GPUS ${num_gpu_devices}
                ЛОГ_ПЕРИОД 1"
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u инструменты/run_net.py "${train_cmd}
    fi
    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 300
    kill -9 ${train_pid}
}
source ${BENCHMARK_ROOT}/comparision_system/common_scripts/run_model.sh
_set_params $@
_run