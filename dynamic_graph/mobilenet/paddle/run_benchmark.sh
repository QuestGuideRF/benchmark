#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 sp|mp 1000(max_iter) имя_модели(MobileNetV1|MobileNetV2)"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=128
    if [ ${4} != "MobileNetV1" ] && [ ${4} != "MobileNetV2" ]; then
        echo "------------> пожалуйста, проверьте название модели!"
        exit 1
    fi
    model_name=${4}
    run_mode=${2:-"sp"}
    max_iter=${3:-"1000"}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    direction_id=0
    mission_name="图像分类"
    skip_steps=5
    keyword="batch_cost:"
    separator=" "
    position=11
    model_mode=0
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}
function _train(){
    train_cmd="--batch_size=${base_batch_size} \
               --total_images=1281167 \
               --class_dim=1000 \
               --image_shape=3,224,224 \
               --model_save_dir=выход \
               --lr_strategy=piecewise_decay \
               --lr=0.1 \
               --data_dir=./data/ILSVRC2012_Pytorch/dataset_100/ \
               --l2_decay=3e-5 \
               --model=${model_name} \
               --max_iter=${max_iter} \
               --num_epochs=2 "
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u поезд.py "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m Paddle.distributed.launch --log_dir=./mylog train.py --use_data_parallel=1 "${train_cmd}
        log_parse_file="mylog/workerlog.0"
    fi
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_name}, УСПЕХ"
        export job_fail_flag=0
    fi
    if [ ${run_mode} != "sp"  -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run