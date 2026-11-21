#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 sp|mp 1000(max_iter)"
    exit
fi
function _set_params(){
    index="$1"
    run_mode=${2:-"sp"}
    max_iter=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    base_batch_size=32
    model_name="STGAN"_bs${base_batch_size}
    mission_name="图像生成"
    direction_id=0
    skip_steps=5
    keyword="ips:"
    ips_unit="images/s"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}
function _set_env(){
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_conv_workspace_size_limit=256
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, package_size=${base_batch_size}"
    train_cmd=" --model_net ${model_name%_bs*} \
        --dataset знаменитости \
        --crop_size 170 \
        --image_size 128 \
        --train_list ./data/celeba/list_attr_celeba.txt \
        --gan_mode wgan \
        --batch_size ${base_batch_size} \
        --print_freq 5 \
        --num_discriminator_time 1 \
        --epoch 50 \
        --profile=${is_profiler} \
        --profiler_path=${profiler_path} \
        --max_iter=${max_iter} \
        --run_test Ложь"
    case ${run_mode} in
    sp)
        train_cmd="python -u поезд.py "${train_cmd}
        ;;
    mp)
        rm -rf ./mylog
        train_cmd="python -m Paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0"
        ;;
    *)
        echo "выберите run_mode: sp или mp"
        exit 1
        ;;
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
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run