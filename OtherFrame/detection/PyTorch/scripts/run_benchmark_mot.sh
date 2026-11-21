#!/usr/bin/env bash
set -x
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"2"}
    fp_item=${3:-"fp32"}
    max_epoch=${4:-"1"}
    model_name=${5:-"model_name"}
    run_log_path=$(pwd)
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
}
function _analysis_log(){
    python3.7 analysis_log_mot.py \
                --filename ${log_file} \
                --jsonname ${res_log_file} \
                --keyword "time:" \
                --model_name ${model_name}_bs${batch_size}_${fp_item} \
                --run_mode ${run_mode} \
                --gpu_num ${num_gpu_devices} \
                --batch_size ${batch_size}
    cp ${res_log_file} /workspace
}
function _train(){
    echo "Обучите ${model_name} на ${num_gpu_devices} графических процессорах"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    log_iter="2"
    if [ ${model_name} = "jde" ]; then
        optimizer_lr="0.01"
        train_cmd="python3.7 -u train.py --batch-size ${batch_size} --epochs ${max_epoch} --lr ${optimizer_lr} --print-interval ${log_iter}"
    else
        optimizer_lr="0.0001"
        if [ ${num_gpu_devices} = "1" ]; then
            gpus_info='0'
        else
            gpus_info='0,1,2,3,4,5,6,7'
        fi
        train_cmd="python3.7 -u train.py --batch_size=${batch_size} --num_epochs=${max_epoch} --lr=${optimizer_lr} --print_iter=${log_iter} --gpus=${gpus_info}"
    fi
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_name}, УСПЕХ"
        export job_fail_flag=0
    fi
    _analysis_log
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}
_set_params $@
_train