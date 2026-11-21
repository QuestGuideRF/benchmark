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
    log_file=${run_log_path}/${model_name}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/${model_name}_${fp_item}_${num_gpu_devices}_speed
}
function _analysis_log(){
    python analysis_log.py ${model_name} ${batch_size} ${log_file} ${res_log_file}
    cp ${log_file} /workspace
    cp ${res_log_file} /workspace
}
function _train(){
    echo "Обучите ${model_name} на ${num_gpu_devices} графических процессорах"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    train_config="mmedi_benchmark_configs/${model_name}.py"
    train_options="--no-validate "
    case ${run_mode} in
    sp) train_cmd="./tools/dist_train.sh ${train_config} 1 ${train_options}" ;;
    mp)
        case ${model_name} in
        basicvsr_mp_bs2|basicvsr_mp_bs4) train_cmd="./tools/dist_train.sh ${train_config} 4 ${train_options}" ;;
        *) train_cmd="./tools/dist_train.sh ${train_config} 8 ${train_options}"
        esac
        ;;
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
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}
_set_params $@
_train