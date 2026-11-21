#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3 big|medium|small static|padding sp|mp 3(max_epoch)"
    exit
fi
function _set_params(){
    index=$1
    model_type=$2
    rnn_type=$3
    run_mode=${4:-"sp"}
    max_epoch=${5}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    if [[ ${index} -eq 6 ]]; then base_batch_size=12000; else base_batch_size=20; fi
    model_name="paddingrnn_"${model_type}_${rnn_type}_bs${base_batch_size}
    mission_name="语言模型"
    direction_id=1
    skip_steps=1
    keyword="avg_time:"
    separator=" "
    position=8
    model_mode=1
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}
function _set_env(){
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export FLAGS_fraction_of_gpu_memory_to_use=1.0
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_memory_fraction_of_eager_deletion=0.5
    if [[ ${model_type} == "large" ]]; then
        export FLAGS_memory_fraction_of_eager_deletion=1.0
    fi
}
function _train(){
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    python -c "import paddle; print(paddle.__version__)"
    train_cmd="--data_path к данным/простые-примеры/данные/ \
      --model_type $model_type \
      --use_gpu Истина \
      --enable_ce \
      --max_epoch=${max_epoch} \
      --rnn_model ${rnn_type} \
      --use_dataloader Истина \
      --enable_auto_fusion Истина \
      --profile ${is_profiler} \
      --profiler_path=${profiler_path} \
      --batch_size ${batch_size}"
     timeout 15m python -u train.py ${train_cmd} > ${log_file} 2>&1
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