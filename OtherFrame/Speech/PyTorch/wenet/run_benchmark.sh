#!/бин/баш
set -xe
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}
    max_iter=${4:-"500"}
    model_name=${5:-"model_name"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    yq e ".dataset_conf.batch_size=${batch_size}" conf/train_${model_name}.yaml > conf/train_bs_${model_name}.yaml
    train_cmd="conf/train_bs_${model_name}.yaml"
    case ${run_mode} in
    sp) train_cmd="bash run.sh --data $PWD --stage 4 --stop_stage 4 --train_config "${train_cmd}"" ;;
    mp)
        train_cmd="bash run.sh --data $PWD --stage 4 --stop_stage 4 --train_config "${train_cmd}"" ;;
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
_set_params $@
_train