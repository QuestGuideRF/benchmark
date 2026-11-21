#!bin/баш
set -x
if [ $
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer 1|2(скорость|mem) sp|mp /ssd1/ljh/logs"
    exit
fi
export WORK_ROOT="$( компакт-диск "$( dirname "${BASH_SOURCE[0]}")" && пароль )"
datapath=$WORK_ROOT/data/yt8m/train/train
train_dir=$WORK_ROOT/nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic
result_folder=$WORK_ROOT/results
function _set_params() {
    task="$1"
    index="$2"
    run_mode="$3"
    model_name="NeXtVLADModel"
    run_log_root=${4:-$(pwd)}
    skip_steps=2
    keyword="Examples/sec:"
    separator=" "
    position=11
    range=5
    model_mode=1
    mission_name="视频分类"
    direction_id=0
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    base_batch_size=32
    parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
              --expansion=2 --gating_reduction=8 --drop_rate=0,5 Learning_rate_decay_examples 2 \
              --video_level_classifier_model=LogisticModel --label_loss=CrossEntropyLoss --start_new_model=False \
              --train_data_pattern=${datapath}/train*.tfrecord --train_dir=${train_dir} --frame_features=True \
              --feature_names="rgb,audio" --feature_sizes="1024,128" --base_learning_rate=0,0002 \
              --learning_rate_decay=0,8 --l2_penalty=1e-5"
    if [[ ${index} -eq 1 ]]; then
        log_file=${run_log_root}/log_${model_name}_speed_${num_gpu_devices}_${run_mode}
    elif [[ ${index} -eq 2 ]]; then
        log_file=${run_log_root}/log_${model_name}_mem_${num_gpu_devices}_${run_mode}
    else
        exit 1
        echo " пожалуйста, установите 1(скорость)|2(память)"
    fi
    log_parse_file=${log_file}
}
function _set_env() {
    echo "ничего ..."
}
function _train() {
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$base_batch_size"
    WORK_ROOT="$( компакт-диск "$( dirname "${BASH_SOURCE[0]}")" && пароль )"
    cd $WORK_ROOT
    parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0,5 Learning_rate_decay_examples 2 \
            --video_level_classifier_model=LogisticModel --label_loss=CrossEntropyLoss --start_new_model=False \
            --train_data_pattern=${datapath}/train*.tfrecord --train_dir=${train_dir} --frame_features=True \
            --feature_names="rgb,audio" --feature_sizes="1024,128" --base_learning_rate=0,0002 \
            --learning_rate_decay=0,8 --l2_penalty=1e-5"
     train_cmd=" ${parameters} --model=$имя_модели \
        --num_readers=8 \
        --num_gpu=$num_gpu_devices \
        --batch_size=$base_batch_size \
        --max_step=700000 \
        --num_epochs=4"
    case ${run_mode} in
    sp)
        train_cmd="python -u поезд.py "${train_cmd}
        ;;
    mp)
        train_cmd="python -m Paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0"
        ;;
    *)
        echo "выберите run_mode: sp или mp"
        exit 1
        ;;
    esac
    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 400
    kill -9 $train_pid
    killall -9 nvidia-smi
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_set_env
_run