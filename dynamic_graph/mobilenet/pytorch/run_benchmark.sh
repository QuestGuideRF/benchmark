#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 sp|mp 1(num_epoch) имя_модели(MobileNetV1|MobileNetV2|resnet)"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=128
    model_name=${4}
    if [ ${model_name} != "MobileNetV1" ] && [ ${model_name} != "MobileNetV2" ] && [ ${model_name} != "resnet" ]; then
        echo "------------> пожалуйста, проверьте название модели!"
        exit 1
    fi
    run_mode=${2}
    num_epoch=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    direction_id=0
    mission_name="图像分类"
    skip_steps=5
    keyword="batch_cost:"
    separator=" "
    position=9
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
    if [ ${model_name} = "resnet" ]; then
        model_name_arg="ResNet50"
        num_workers=8
        data_dir="./data/ILSVRC2012"
    else
        model_name_arg=${model_name}
        num_workers=10
        data_dir="./data/ILSVRC2012_Pytorch/dataset_100"
    fi
    train_cmd="--batch_size=${base_batch_size} \
               --workers=${num_workers} \
               --data_dir=${data_dir} \
               --model=${model_name_arg} \
               --epochs=${num_epoch}"
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u train_pytorch.py "${train_cmd}
    else
        train_cmd="python3 -m torch.distributed.launch --nproc_per_node=${num_gpu_devices} --use_env train_pytorch.py "${train_cmd}
    fi
    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 1200
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run