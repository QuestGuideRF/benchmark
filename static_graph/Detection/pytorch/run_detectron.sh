#!bin/баш
set -x
if [[ $
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 скорость|mem|maxbs имя_модели sp /ssd1/ljh/logs"
    echo "имя_модели может быть:"
    echo "  маска_rcnn_fpn_resnet, маска_rcnn_fpn_resnext, retinanet_rcnn_fpn, cascade_rcnn_fpn"
    exit
fi
function _set_params(){
    index=$1
    base_batch_size=2
    model_name=$2
    run_log_path=${4:-$(pwd)}
    run_mode=${3:-"sp"}
    skip_steps=2
    keyword="json_stats:"
    separator=""
    direction_id=0
    mission_name="目标检测"
    model_mode=0
    position=-1
    if [[ ${model_name} = "cascade_rcnn_fpn" ]]; then
        range=-1
    else
        range=1:9
    fi
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${
    if [[ ${index} = "analysis" ]]; then
        log_file=${run_log_path}/${model_name}_speed_${num_gpu_devices}_${run_mode}
    else
        log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    fi
    log_parse_file=${log_file}
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    WORK_ROOT="$( компакт-диск "$( dirname "${BASH_SOURCE[0]}" )" && пароль )"
    if [[ ${model_name} = "cascade_rcnn_fpn" ]]; then
        cp -f ${WORK_ROOT}/Detectron/detectron/utils/io.py ${WORK_ROOT}/Detectron-Cascade-RCNN/detectron/utils/io.py
        cp -f ${WORK_ROOT}/Detectron/detectron/utils/env.py ${WORK_ROOT}/Detectron-Cascade-RCNN/detectron/utils/env.py
        grep -q "По адресу Detectron-Cascade-RCNN/detectron/utils/cython_bbox.pyx" ${WORK_ROOT}/Detectron-Cascade-RCNN/tools/test_net.py
        if [ $? -eq 0 ]; then
            echo "----------файлы .pyx уже обработаны"
        else
            sed -i "33 a ##### По адресу Detectron-Cascade-RCNN/detectron/utils/cython_bbox.pyx" ${WORK_ROOT}/Detectron-Cascade-RCNN/tools/test_net.py
            sed -i "34 импорт pyximport" ${WORK_ROOT}/Detectron-Cascade-RCNN/tools/test_net.py
            sed -i "35 pyximport.install()" ${WORK_ROOT}/Detectron-Cascade-RCNN/tools/test_net.py
        fi
        DETECTRON_REPO_NAME=Detectron-Cascade-RCNN
        train_cmd="--cfg configs/e2e_cascade_rcnn_R-50-FPN_1x.yaml OUTPUT_DIR ./output"
    elif [[ ${model_name} = "mask_rcnn_fpn_resnet" ]]; then
        DETECTRON_REPO_NAME=Detectron
        train_cmd="--cfg configs/e2e_mask_rcnn_R-101-FPN_1x.yaml OUTPUT_DIR ./output"
    elif [[ ${model_name} = "mask_rcnn_fpn_resnext" ]];then
        DETECTRON_REPO_NAME=Detectron
        train_cmd="--cfg configs/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml OUTPUT_DIR ./output"
    elif [[ ${model_name} = "retinanet_rcnn_fpn" ]];then
        DETECTRON_REPO_NAME=Detectron
        train_cmd="--cfg configs/retinanet_R-50-FPN_1x.yaml OUTPUT_DIR ./output"
    else
        echo "имя_модели должно быть Mask_rcnn_fpn_resnet | маска_rcnn_fpn_resnext | retinanet_rcnn_fpn | cascade_rcnn_fpn"
        exit 1
    fi
    export PYTHONPATH=${WORK_ROOT}/${DETECTRON_REPO_NAME}:$PYTHONPATH
    cp -f ${WORK_ROOT}/src/dataset_catalog.py ${WORK_ROOT}/${DETECTRON_REPO_NAME}/detectron/datasets/dataset_catalog.py
    cd ${WORK_ROOT}/${DETECTRON_REPO_NAME}
    python setup.py install --user
    make
    cd ${WORK_ROOT}
    python -c 'import torch; print(torch.__version__)'
    train_cmd="python -u ${DETECTRON_REPO_NAME}/tools/train_net.py "${train_cmd}
    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    if [[ ${model_name} = "mask_rcnn_fpn_resnext" ]];then
        sleep 900
    else
        sleep 600
    fi
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}
source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_run