#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 Pix2pix sp|mp 1000(max_iter)"
    exit
fi
function _set_params(){
    index="$1"
    model_name="$2"
    run_mode=${3:-"sp"}
    max_iter=${4}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    mission_name="图像生成"
    direction_id=0
    skip_steps=5
    keyword="ips:"
    ips_unit="images/s"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    if echo {DCGAN CGAN} | grep -w $model_name &>/dev/null
    then
        base_batch_size=32
    elif echo {Pix2pix} | grep -w $model_name &>/dev/null
    then
        base_batch_size=1
    else
        echo "модель: $model_name не поддерживается!"
    fi
    model_name=${model_name}_bs${base_batch_size}
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}
function _set_env(){
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_eager_delete_tensor_gb=0.0
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices"
    if echo {DCGAN CGAN} | grep -w ${model_name%_bs*} &>/dev/null
    then
        echo "${model_name}, размер_пакета: ${base_batch_size}"
        train_cmd=" --model_net ${model_name%_bs*} \
           --dataset mnist \
           --noise_size 100 \
           --batch_size ${base_batch_size} \
           --epoch 10 \
           --profile=${is_profiler} \
           --profiler_path=${profiler_path} \
           --max_iter=${max_iter}"
    elif echo {Pix2pix} | grep -w ${model_name%_bs*} &>/dev/null
    then
        echo "${model_name}, размер_пакета: ${base_batch_size}"
        train_cmd=" --model_net ${model_name%_bs*} \
           --dataset городские пейзажи \
           --train_list data/cityscapes/pix2pix_train_list \
           --test_list data/cityscapes/pix2pix_test_list \
           --crop_type Случайный \
           --dropout Истина \
           --gan_mode ванильный \
           --batch_size ${base_batch_size} \
           --эпоха 200 \
           --crop_size 256 \
           --profile=${is_profiler} \
           --profiler_path=${profiler_path} \
           --max_iter=${max_iter}"
    fi
    train_cmd="python -u поезд.py "${train_cmd}
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_name}, УСПЕХ"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run