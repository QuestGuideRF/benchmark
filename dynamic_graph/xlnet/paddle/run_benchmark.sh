#!/usr/bin/env bash
set -xe
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}
    max_iter=${4:-"200"}
    model_item=${5:-"xlnet-base-cased"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    need_profile=${6:-"off"}
    base_batch_size=${batch_size}
    mission_name="语义表示"
    direction_id=1
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    model_name=${model_item}_bs${batch_size}_${fp_item}
    log_with_profiler=$log_file
    profiler_path=$log_profile
    keyword="avg_batch_cost:"
    keyword_loss=""
    separator=""
    position=""
    range=""
    skip_steps=20
    model_mode=0
    ips_unit='sequences/s'
    index="1"
    gpu_num=$num_gpu_devices
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    profiler_cmd=""
    profiler_options="пакетный_диапазон=[100,110];profile_path=${log_profile}"
    if [ $need_profile = "on" ]; then
        profiler_cmd="--profiler_options=${profiler_options}"
    fi
    train_cmd="${profiler_cmd}
               --model_name_or_path=${model_item}
               --task_name=SST-2
               --max_seq_length=128
               --pad_to_max_seq_len=Истина
               --logging_steps=1
               --save_steps=2000
               --batch_size=${batch_size}
               --learning_rate=2e-5
               --max_steps=${max_iter}
               --output_dir=${run_log_path}
    "
    case ${run_mode} in
    sp)
        train_cmd="python -m Paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES \
        примеры/language_model/xlnet/run_glue.py ${train_cmd}" ;;
    mp)
        train_cmd="python -m Paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES \
        примеры/language_model/xlnet/run_glue.py ${train_cmd}"
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
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run