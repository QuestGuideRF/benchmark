#!/usr/bin/env bash
set -xe
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}
    model_item=${4:-"model_item"}
    config_path=${5:-"config_path"}
    run_log_path=$(pwd)
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}.log
    case ${model_item} in
    MobileNetV2) model_name=MobileNetV2;;
    ShuffleNetV2) model_name=ShuffleNetV2_x1_0;;
    SwinTransformer)  model_name=SwinTransformer_tiny_patch4_window7_224;;
    MobileNetV3Large1.0) model_name=MobileNetV3_large_x1_0;;
    *) echo "установите свою модель с помощью MobileNetV2|ShuffleNetV2|SwinTransformer|MobileNetV3Large1.0"; exit 1;
    esac
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    case ${run_mode} in
    sp) train_cmd="python Tools/train.py ${config_path} --no-validate --cfg-options data.samples_per_gpu=${batch_size} log_config.interval=10";;
    mp)
	train_cmd="python -m torch.distributed.launch --nproc_per_node=${num_gpu_devices} --master_port=29500 ./tools/train.py ${config_path} --no-validate --cfg-options data.samples_per_gpu=${batch_size} log_config.interval=10 --launcher pytorch";;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    timeout 15m ${train_cmd}  > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_item}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_item}, УСПЕХ"
        export job_fail_flag=0
    fi
python analysis_log.py -d work_dirs -m ${model_name} -b ${batch_size} -n ${num_gpu_devices}
eval "mv work_dirs ${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}"
}
_set_params $@
rm -rf work_dirs
source ${ROOT_DIR}/scripts/run_model.sh
_run