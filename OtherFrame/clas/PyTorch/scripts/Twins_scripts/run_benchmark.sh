#!/usr/bin/env bash
set -xe
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}
    epoch=${4:-"300"}
    model_item=${5:-"model_item"}
    run_log_path=$(pwd)
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}.log
    model_name=alt_gvt_base
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    train_cmd="--model ${model_item} --batch-size ${batch_size} --data-path data/imagenet --dist-eval --drop-path 0.3 --epochs ${epoch}"
    case ${run_mode} in
    sp) train_cmd="python main.py ${train_cmd}" ;;
    mp)
        train_cmd="python -m torch.distributed.launch --nproc_per_node=${num_gpu_devices} --use_env main.py ${train_cmd}" ;;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    timeout 15m ${train_cmd}  > ${log_file} 2>&1
    python analysis_log.py -f ${log_file} -m ${model_item} -b ${batch_size} -n ${num_gpu_devices}
    if [ $? -ne 0 ];then
        echo -e "${model_item}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_item}, УСПЕХ"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}
_set_params $@
source ${ROOT_DIR}/scripts/run_model.sh
_run