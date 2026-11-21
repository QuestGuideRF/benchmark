#!bin/баш
set -x
if [[ $
    echo "dict текущего задания: {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1 sp имя_модели(cyclegan|pix2pix)"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=1
    run_mode=${2:-"sp"}
    model_name=$3
    if [ ${3} != "cyclegan" ] && [ ${3} != "pix2pix" ]; then
        echo "------------> пожалуйста, проверьте название модели! это должно быть циклоган|pix2pix"
        exit 1
    fi
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    mission_name="图像生成"
    direction_id=0
    skip_steps=5
    keyword="time:"
    separator=" "
    position=5
    range=0:4
    model_mode=0
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}
}
function _train(){
    if [ ${model_name} = "cyclegan" ];then
	train_cmd="--dataroot ./datasets/cityscapes \
		   --name городские пейзажи_cyclegan \
		   --model цикл_ган \
		   --pool_size 50 \
		   --no_dropout\
		   --display_id 0"
    else
        train_cmd="--dataroot ./datasets/cityscapes \
		   --name городские пейзажи_pix2pix \
		   --model pix2pix \
		   --netG unet_256 \
		   --направление БтоА \
		   --lambda_L1 100 \
		   --dataset_mode выровнен \
		   --нормальная партия \
		   --pool_size 0 \
		   --display_id 0"
    fi
    train_cmd="python -u поезд.py "${train_cmd}
    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 300
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run