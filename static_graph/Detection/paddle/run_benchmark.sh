#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3|6 имя_модели sp|mp 1000(max_iter)"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=2
    model_name=$2
    mission_name="目标检测"
    direction_id=0
    run_mode=${3:-"sp"}
    max_iter=${4}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    skip_steps=2
    keyword="ips:"
    model_mode=-1
    ips_unit="images/s"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    if [ ${model_name} = "mask_rcnn_fpn_resnet" ] || [ ${model_name} = "cascade_rcnn_fpn" ]; then
        base_learning_rate=0.002
    elif [ ${model_name} = "mask_rcnn_fpn_resnext" ] || [ ${model_name} = "retinanet_rcnn_fpn" ]; then
        base_learning_rate=0.001
    else
        echo "имя_модели должно быть Mask_rcnn_fpn_resnet | маска_rcnn_fpn_resnext | retinanet_rcnn_fpn | cascade_rcnn_fpn"
        exit 1
    fi
    if [[ ${run_mode} = "sp" ]]; then
        batch_size=`expr $base_batch_size \* $num_gpu_devices`
        learning_rate=$(awk 'НАЧАЛО { печать '${base_learning_rate}' * '${num_gpu_devices}' }')
    else
        batch_size=$base_batch_size
        learning_rate=${base_learning_rate}
    fi
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
    if [[ ${model_name} = "mask_rcnn_fpn_resnet" ]]; then
        config_file="configs/mask_rcnn_r101_vd_fpn_1x.yml"
    elif [[ ${model_name} = "mask_rcnn_fpn_resnext" ]];then
        config_file="configs/mask_rcnn_x101_vd_64x4d_fpn_1x.yml"
    elif [[ ${model_name} = "retinanet_rcnn_fpn" ]];then
        config_file="configs/retinanet_r50_fpn_1x.yml"
    elif [[ ${model_name} = "cascade_rcnn_fpn" ]];then
        config_file="configs/cascade_rcnn_r50_fpn_1x.yml"
    else
        echo "имя_модели должно быть Mask_rcnn_fpn_resnet | маска_rcnn_fpn_resnext | retinanet_rcnn_fpn | cascade_rcnn_fpn"
        exit 1
    fi
    model_name=${model_name}_bs${base_batch_size}
}
function _set_env(){
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    export FLAGS_conv_workspace_size_limit=4096
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    train_cmd="-c ${config_file} \
               --opt LearningRate.base_lr=${learning_rate} TrainReader.batch_size=${base_batch_size} max_iters=${max_iter} \
               --is_profiler=${is_profiler} \
               --profiler_path=${profiler_path}"
    case ${run_mode} in
    sp) train_cmd="python -u инструменты/train.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog
        train_cmd="python -m Paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES инструменты/train.py "${train_cmd}
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
    kill -9 `ps -ef|grep python |awk '{print $2}'`
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run