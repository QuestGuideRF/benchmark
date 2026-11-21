#!/бин/баш
set -x
if [ $
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_image_resnet.sh скорость|mem|maxbs 32 resnet50|resnet101 sp|mp /ssd1/ljh/logs"
    exit
fi
function _set_params() {
    index=$1
    base_batch_size=$2
    model_name=$3
    run_mode="sp"
    run_log_root=${5:-$(pwd)}
    mission_name="图像分类"
    direction_id=0
    skip_steps=2
    keyword="time:"
    separator=" "
    position=29
    model_mode=0
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    batch_size=`expr $base_batch_size \* $num_gpu_devices`
    num_workers=`expr 8 \* $num_gpu_devices`
    if [[ ${index} = "analysis" ]]; then
        log_file=${run_log_root}/log_vision_${model_name}_speed_${num_gpu_devices}_${run_mode}
    else
        log_file=${run_log_root}/log_vision_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    fi
    log_parse_file=${log_file}
}
function _train() {
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    WORK_ROOT="$( компакт-диск "$( dirname "${BASH_SOURCE[0]}")" && пароль )"
    data_path="/ssd2/liyang/benchmark/benchmark/static_graph/image_classification/pytorch/ILSVRC2012/"
    num_epochs=2
    python -c "import torch; print(torch.__version__)"
    export PYTHONPATH=${BENCHMARK_ROOT}/third_party/pytorch/vision
    echo "${model_name}, размер_пакета: ${batch_size}"
    stdbuf -oL python ${BENCHMARK_ROOT}/third_party/pytorch/vision/references/classification/train.py \
           --data-path ${data_path} \
           --model ${model_name} \
           --device cuda \
           --batch-size ${batch_size} \
           --epochs ${num_epochs} \
           --workers ${num_workers} \
           --print-freq 10 \
           --output-dir ./output/vision \
           --cache-dataset > ${log_file} 2>&1 &
    train_pid=$!
    sleep 500
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}
source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_run