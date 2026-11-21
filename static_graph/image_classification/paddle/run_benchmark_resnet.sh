#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 32 имя_модели(ResNet50_bs32|ResNet50_bs128|ResNet101|SE_ResNeXt50_32x4d) sp|mp max_epoch"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=$2
    model_name=$3
    run_mode=${4:-"sp"}
    mission_name="图像分类"
    direction_id=0
    max_epoch=${5}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    skip_steps=8
    keyword="ips:"
    keyword_loss="loss:"
    model_mode=-1
    ips_unit="images/s"
    devices=(${CUDA_VISIBLE_DEVICES//,/ })
    num_gpu_devices=${
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
     if [ ${run_mode} == "sp" ]; then
        is_distributed=False
     else
        is_distributed=True
     fi
}
function _set_env(){
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_conv_workspace_size_limit=4000
}
function _train(){
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    WORK_ROOT=$PWD
    echo "${model_name}, размер_пакета: ${batch_size}"
    if [ ${model_name%%_bs*} == "ResNet50" ]; then
        config_file="ppcls/configs/ImageNet/ResNet/ResNet50.yaml"
    elif [ ${model_name%%_bs*} == "ResNet101" ]; then
         config_file="ppcls/configs/ImageNet/ResNet/ResNet101.yaml"
    elif [ ${model_name%%_bs*} == "SE_ResNeXt50_32x4d" ]; then
          config_file="ppcls/configs/ImageNet/SENet/SE_ResNeXt50_32x4d.yaml"
    else
        echo "модель: $model_name не поддерживается!"
        exit
    fi
    if [ ${model_name%%_bs*} == "ResNet50" ]; then
        fuse_elewise_add_act_ops="True"
        enable_addto="True"
        export FLAGS_max_inplace_grad_add=8
    else
        fuse_elewise_add_act_ops="False"
        enable_addto="False"
    fi
    train_cmd="-c $config_file
               -o Global.print_batch_step=10
               -o DataLoader.Train.sampler.batch_size=${base_batch_size}
               -o Global.eval_during_train=False
               -o DataLoader.Train.dataset.image_root=./dataset/imagenet100_data
               -o DataLoader.Train.dataset.cls_label_path=./dataset/imagenet100_data/train_list.txt
               -o плавкий_elewise_add_act_ops=${fuse_elewise_add_act_ops}
               -o Enable_addto=${enable_addto}
               -o DataLoader.Train.loader.num_workers=8
               -o Global.is_distributed=${is_distributed}
               -o Global.epochs=${max_epoch}"
    case ${run_mode} in
    sp) train_cmd="python -u ppcls/static/train.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog_${model_name}
        if [ ${model_name} = "ResNet50_bs32" ] || [ ${model_name} = "ResNet50_bs128" ] || [ ${model_name} = "ResNet50_bs96" ]; then
            export FLAGS_fraction_of_gpu_memory_to_use=0.8
            train_cmd="python -m Paddle.distributed.launch --log_dir=./mylog_${model_name} --gpus=$CUDA_VISIBLE_DEVICES ppcls/static/train.py "${train_cmd}" -o Global.use_dali=Истина"
        else
            train_cmd="python -m Paddle.distributed.launch --log_dir=./mylog_${model_name} --gpus=$CUDA_VISIBLE_DEVICES ppcls/static/train.py "${train_cmd}
        fi
        log_parse_file="mylog_${model_name}/workerlog.0" ;;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    timeout 5m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_name}, УСПЕХ"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep python |awk '{print $2}'`
    if [ $run_mode = "mp" -a -d mylog_${model_name} ]; then
        rm ${log_file}
        cp mylog_${model_name}/workerlog.0 ${log_file}
    fi
    cd ${WORK_ROOT}
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run