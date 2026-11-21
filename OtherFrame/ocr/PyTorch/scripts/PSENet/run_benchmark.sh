#!/usr/bin/env bash
set -xe
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}
    max_epoch=${4:-"10"}
    run_log_path=$(pwd)
    model_name="det_r50_vd_pse"_bs${batch_size}_${fp_item}
    mission_name="OCR"
    direction_id=0
    ips_unit="samples/sec"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    index_log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    rm -rf checkpoints/
    train_cmd="config/psenet/psenet_r50_ic15_736.py --cfg-options data.batch_size=${batch_size} train_cfg.epoch=${max_epoch} model.backbone.pretrained=False"
    case ${run_mode} in
    sp) train_cmd="python3.7 поезд.py "${train_cmd} ;;
    mp)
        train_cmd="python3.7 поезд.py "${train_cmd} ;;
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
}
function _analysis_log(){
    analysis_cmd="python3.7 Analysis_log.py --filename ${log_file} --model_name ${model_name} --mission_name ${mission_name} --run_mode ${run_mode} --direction_id 0 --keyword 'ips:' --base_batch_size ${batch_size} --skip_steps 1 --gpu_num ${num_gpu_devices} --index 1 --model_mode=-1 --ips_unit=images/sec"
    eval $analysis_cmd
}
function _kill_process(){
    kill -9 `ps -ef|grep 'python3.7'|awk '{print $2}'` >/dev/null 2>&1
}
_set_params $@
_train
_analysis_log
_kill_process