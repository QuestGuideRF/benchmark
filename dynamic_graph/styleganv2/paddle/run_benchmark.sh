#!/usr/bin/env bash
set -xe
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}
    mode=${4:-"epochs"}
    max_iter=${5:-"500"}
    model_name=${6:-"model_name"}
    config=${7:-"config"}
    log_interval=${8:-"1"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    need_profile=${9:-"off"}
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
    log_profile=${run_log_path}/${model_name}_model.profile
    base_batch_size=${batch_size}
    mission_name="视频分类"
    direction_id="0"
    ips_unit="samples/sec"
    skip_steps=10
    keyword="ips:"
    keyword_loss="loss:"
    index="1"
}
function _analysis_log(){
    python benchmark/analysis_log.py ${model_name} ${log_file} ${res_log_file}
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    profiler_cmd=""
    profiler_options="пакетный_диапазон=[10,20];profile_path=${log_profile}"
    if [ $need_profile = "on" ]; then
        profiler_cmd="--profiler_options=${profiler_options}"
    fi
    train_cmd="${profiler_cmd}
               --config-file=${config}
               -o dataset.train.batch_size=${batch_size}
               log_config.interval=${log_interval}
               ${mode}=${max_iter} "
    case ${run_mode} in
    sp) train_cmd="инструменты python -u/main.py "${train_cmd} ;;
    mp)
        train_cmd="python -m Paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES инструменты/main.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_name}, УСПЕХ"
        export job_fail_flag=0
    fi
    trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run