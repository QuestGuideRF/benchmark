#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh 1|2 sp|mp /ssd1/ljh/logs"
    exit
fi
function _set_params(){
    index="$1"
    run_mode=${2:-"sp"}
    run_log_path=${3:-$(pwd)}
    model_name="AttGAN"
    mission_name="图像生成"
    direction_id=0
    skip_steps=5
    keyword="Batch_time_cost:"
    separator=" "
    position=-1
    model_mode=0
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    base_batch_size=32
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}
}
function _set_env(){
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_conv_workspace_size_limit=256
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    train_cmd=" --model_net $имя_модели \
        --dataset знаменитости \
        --crop_size 170 \
        --image_size 128 \
        --train_list ./data/celeba/list_attr_celeba.txt \
        --gan_mode wgan \
        --batch_size $base_batch_size \
        --print_freq 5 \
        --num_discriminator_time 5 \
        --epoch 120 \
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
    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 300
    kill -9 $train_pid
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run