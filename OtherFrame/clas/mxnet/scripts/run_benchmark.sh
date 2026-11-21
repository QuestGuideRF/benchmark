#!/usr/bin/env bash
set -xe
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}
    epoch=${4:-"300"}
    model_name=${5:-"model_name"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    train_cmd="python scripts/classification/imagenet/train_imagenet.py --num-epochs ${epoch} --batch-size ${batch_size} --mode Hybrid --num-gpus ${num_gpu_devices} --data-dir ./data --model ${model_name}"
    case ${run_mode} in
    sp) train_cmd="${train_cmd}" ;;
    mp)
        train_cmd="${train_cmd} -j 32" ;;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    timeout 15m ${train_cmd} 2>&1|tee ${log_file}
    python analysis_log.py -f ${log_file} -m ${model_name} -b ${batch_size} -n ${num_gpu_devices}
    if [ $? -ne 0 ];then
        echo -e "${model_name}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_name}, УСПЕХ"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}
sed -i '{s/batch\[j\]/batch/g}' scripts/classification/imagenet/train_imagenet.py
_set_params $@
_train