#!/usr/bin/env bash
set -xe
function _set_params(){
    model_name=${1:-"model_name"}
    run_mode=${2:-"sp"}
    fp_item=${3:-"fp32"}
    batch_size=${4:-"2"}
    max_iter=${5:-"100"}
    num_workers=${6:-"3"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
}
function _analysis_log(){
    python analysis_log.py ${model_name} ${log_file} ${res_log_file}
    cp ${res_log_file} /workspace
}
function _train(){
    echo "Обучите ${model_name} на ${num_gpu_devices} графических процессорах"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    case ${run_mode} in
    sp) train_cmd="python train.py --batch ${batch_size} данные/процесс --iter ${max_iter}" ;;
    mp)
        train_cmd="python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 train.py --batch ${batch_size} data/process --iter ${max_iter}" ;;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_name}, УСПЕХ"
        export job_fail_flag=0
    fi
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
    _analysis_log
    trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM
}
_set_params $@
_train