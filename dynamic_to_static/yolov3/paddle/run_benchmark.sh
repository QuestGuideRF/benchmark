#!bin/баш
set -xe
if [[ $
    echo "dict текущего задания: {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3|6 sp|mp 1000(max_iter)"
    exit
fi
function _set_params(){
    index="$1"
    run_mode=${2:-"sp"}
    max_iter=${3}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    model_name="yolov3"
    mission_name="目标检测"
    direction_id=0
    skip_steps=5
    keyword="ips:"
    model_mode=-1
    ips_unit="images/s"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    base_batch_size=8
    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}
function _set_env(){
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    export FLAGS_memory_fraction_of_eager_deletion=1.0
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$base_batch_size"
    grep -q "#Для адреса max_iter" ppdet/engine/trainer.py
    if [ $? -eq 0 ]; then
        echo "----------уже адресовано max_iter"
    else
        sed -i '/for step_id, data in enumerate(self.loader):/i\            max_step_id = '${max_iter}' #Для адреса max_iter' ppdet/engine/trainer.py
        sed -i '/for step_id, data in enumerate(self.loader):/a\                if step_id == max_step_id: return' ppdet/engine/trainer.py
    fi
    model_name=${model_name}_bs${base_batch_size}
    if [ $num_gpu_devices -eq 1 ]; then norm_type="bn"; else norm_type="sync_bn"; fi
    train_cmd="-c configs/yolov3/yolov3_darknet53_270e_coco.yml --opt epoch=1 TrainReader.batch_size=${base_batch_size} to_static=True worker_num=8norm_type=${norm_type}"
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
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run