#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 32 nextvlad|CTCN sp|mp 2(max_epoch)"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=$2
    model_name=${3}_bs${base_batch_size}
    run_mode=${4:-"sp"}
    max_epoch=${5}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    mission_name="视频分类"
    direction_id=0
    keyword="ips:"
    skip_steps=1
    model_mode=-1
    ips_unit="images/s"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    if [ $run_mode = "sp" ]; then
        batch_size=`expr $base_batch_size \* $num_gpu_devices`
    else
        batch_size=$base_batch_size
    fi
    config_file_name="nextvlad.yaml"
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}
function _set_env(){
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
}
function _train(){
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    WORK_ROOT=$PWD
    echo "${model_name}, размер_пакета: ${batch_size}"
    sed -i "s/num_gpus: [1-8]/num_gpus: ${num_gpu_devices}/g" ./configs/${config_file_name}
    train_cmd=" --model_name ${model_name%_bs*} \
        --config ./configs/${config_file_name} \
        --valid_interval 1 \
        --log_interval 10 \
        --batch_size=$batch_size \
        --is_profiler=${is_profiler} \
        --profiler_path=${profiler_path} \
        --epoch=${max_epoch}"
    case ${run_mode} in
    sp) train_cmd="python -u поезд.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog
        train_cmd="python -m Paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
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
    kill -9 `ps -ef|grep python |awk '{print $2}'`
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
    cd ${WORK_ROOT}
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run