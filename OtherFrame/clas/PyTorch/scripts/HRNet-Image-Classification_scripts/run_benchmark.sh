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
    model_name=HRNet_W48_C
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    sed -i "s/BATCH_SIZE_PER_GPU: 32/BATCH_SIZE_PER_GPU: $batch_size/g" experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
    train_cmd="инструменты Python/train.py --cfg эксперименты/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
    case ${run_mode} in
    sp) sed -ie 'с/GPUS: (0,1,2,3)/GPUS: (0,)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml;;
    mp)
	sed -ie 's/GPUS: (0,1,2,3)/GPUS: (0,1,2,3,4,5,6,7)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
	sed -i 'с/РАБОТНИКИ: 4/РАБОТНИКИ: 32/г' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml;;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    timeout 15m ${train_cmd}  > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_name}, УСПЕХ"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    case ${run_mode} in
    sp) sed -ie 'с/GPUS: (0,)/GPUS: (0,1,2,3)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml;;
    mp)
	sed -ie 's/GPUS: (0,1,2,3,4,5,6,7)/GPUS: (0,1,2,3)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
	sed -i 'с/РАБОТНИКИ: 32/РАБОТНИКИ: 4/г' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml;;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    sed -i "s/BATCH_SIZE_PER_GPU: $batch_size/BATCH_SIZE_PER_GPU: 32/г" experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
}
_set_params $@
rm -rf output
sed -i 's/view/reshape/g' lib/core/evaluate.py
sed -i 'с/PRINT_FREQ: 1000/PRINT_FREQ: 10/г' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
source ${ROOT_DIR}/scripts/run_model.sh
_run
python analysis_log.py -d output -m ${model_name} -b ${batch_size} -n ${num_gpu_devices}