#!bin/баш
set -x
if [[ $
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_cyclegan.sh 1|2(скорость|mem) /log/path"
    exit
fi
function _set_params(){
    index="$1"
    run_log_path=${2:-$(pwd)}
    model_name="CycleGAN"
    skip_steps=3
    keyword="Стоимость времени:"
    separator=" "
    position=-1
    model_mode=0
    mission_name="图像生成"
    direction_id=0
    run_mode=sp
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    base_batch_size=1
    log_file=${run_log_path}/log_${model_name}_${index}_${num_gpu_devices}
}
function _set_env(){
    echo "nothing"
}
function _train(){
    if [ $index -eq 2 ];then
         sed -i '56c TEST_GPU_MEM = Истина' main.py
    elif [ $index -eq 1 ];then
         sed -i '56c TEST_GPU_MEM = Ложь' main.py
    fi
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, package_size=${base_batch_size}"
    python main.py > ${log_file} 2>&1 &
    train_pid=$!
    sleep 300
    kill -9 $train_pid
}
source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_set_env
_run