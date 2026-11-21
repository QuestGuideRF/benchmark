#!bin/баш
set -xe
echo "CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp|mp 128 1(max_epoch) run_log_path"
function _set_params(){
    index="$1"
    run_mode=${2}
    base_batch_size=${3}
    max_epoch=${4}
    model_name="ResNet50_bs128_fp16"
    run_log_path=${5}
    skip_steps=10
    keyword="samples/sec"
    separator=" "
    position=5
    model_mode=1
    run_mode="sp"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    echo $arr
    num_gpu_devices=${
    log_file=${run_log_path}/log_${model_name}_${index}_${num_gpu_devices}_${run_mode}
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    train_cmd="--data-root ./data/ILSVRC2012_mxnet_rec/
               -н ${num_gpu_devices}
	       -б ${base_batch_size}
	       -е ${max_epoch}
	       --dtype float16"
     ./runner ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    if [ ${num_gpu_devices} = 1 ]; then
       sleep 300
   elif [ ${num_gpu_devices} = 8 ]; then
       sleep 500
   else
       sleep 500
   fi
    kill -9 $train_pid
    kill -9 `ps -ef|grep 'runner'|awk '{print $2}'`
}
source ${BENCHMARK_ROOT}/comparision_system/common_scripts/run_model.sh
_set_params $@
_run