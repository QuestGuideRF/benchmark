#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 128 sp|mp 1(max_epoch) режим"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=$2
    mode=${5}
    model_name="ResNet50_bs${base_batch_size}_${mode}_fp16"
    run_mode=${3:-"sp"}
    mission_name="图像分类"
    direction_id=0
    max_epoch=${4}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    skip_steps=8
    keyword="ips:"
    model_mode=-1
    ips_unit="images/s"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
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
    export FLAGS_conv_workspace_size_limit=1500
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_cudnn_batchnorm_spatial_persistent=1
    export FLAGS_fraction_of_gpu_memory_to_use=0.8
}
function _train(){
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    if [ ${mode} == "amp" ]; then
        level=O1
        config_file="ppcls/configs/ImageNet/ResNet/ResNet50_amp_O1_ultra.yaml"
    elif [ ${mode} == "pure" ]; then
        level=O2
	config_file="ppcls/configs/ImageNet/ResNet/ResNet50_amp_O2_ultra.yaml"
    else
        echo "проверьте свой режим!"
    fi
    train_cmd="-с ${config_file}
               -o Global.epochs=${max_epoch}
               -o DataLoader.Train.sampler.batch_size=${base_batch_size}
               -o Global.eval_during_train=False
               -o DataLoader.Train.dataset.image_root=./dataset/imagenet100_data
               -o DataLoader.Train.dataset.cls_label_path=./dataset/imagenet100_data/train_list.txt
               -o DataLoader.Train.loader.num_workers=8
               -o Global.print_batch_step=5
               -o Global.device=gpu
               -o Global.image_shape=[4,224,224]
               -o AMP.level=${level}
               -o Global.is_distributed=${is_distributed}
               "
    case ${run_mode} in
    sp) train_cmd="python -u ppcls/static/train.py "${train_cmd};;
    mp)
        rm -rf ./mylog
        train_cmd="python -m Paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES ppcls/static/train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    timeout 10m ${train_cmd} > ${log_file} 2>&1
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