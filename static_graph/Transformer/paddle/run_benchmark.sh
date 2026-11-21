#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|2|3 sp|mp 100(max_iter) base|big(model_type) fp32|amp_fp16|pure_fp16(fp_mode) bs(4096|2560|5120)"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=$6
    run_mode=${2}
    max_iter=${3}
    model_name="transformer_"${4}
    fp_mode=${5:-"fp32"}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    mission_name="机器翻译"
    direction_id=1
    skip_steps=3
    keyword="ips:"
    model_mode=-1
    ips_unit="words/s"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    log_file=${run_log_path}/${model_name}_bs${base_batch_size}_${fp_mode}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}
function _train(){
    if [ ${model_name} == "transformer_base" ]; then
        config_file="transformer.base.yaml"
    elif [ ${model_name} == "transformer_big" ]; then
        config_file="transformer.big.yaml"
    else
        echo " Модель должна быть Transformer_big или Transformer_base!"
        exit 1
    fi
    if [ ${fp_mode} == "amp_fp16" ]; then
        sed -i "s/^use_amp.*/use_amp: True/g" ../configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: False/g" ../configs/${config_file}
    elif [ ${fp_mode} == "pure_fp16" ]; then
        sed -i "s/^use_amp.*/use_amp: True/g" ../configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: True/g" ../configs/${config_file}
    elif [ ${fp_mode} == "fp32" ]; then
        sed -i "s/^use_amp.*/use_amp: False/g" ../configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: False/g" ../configs/${config_file}
    else
        echo " fp_mode должен быть fp32 pure_fp16 или amp_fp16."
    fi
    sed -i "s/^max_iter.*/max_iter: ${max_iter}/g" ../configs/${config_file}
    sed -i "s/^batch_size:.*/batch_size: ${base_batch_size}/g" ../configs/${config_file}
    model_name=${model_name}_bs${base_batch_size}_${fp_mode}
    train_cmd="--config ../configs/${config_file} --benchmark"
    if [ ${run_mode} = "sp" ]; then
        sed -i "s/^is_distributed.*/is_distributed: False/g" ../configs/${config_file}
        train_cmd="python -u поезд.py "${train_cmd}
    else
        sed -i "s/^is_distributed.*/is_distributed: True/g" ../configs/${config_file}
        rm -rf ./mylog
        train_cmd="python -m Paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES --log_dir ./mylog_${model_name} train.py --distribute "${train_cmd}
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
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run