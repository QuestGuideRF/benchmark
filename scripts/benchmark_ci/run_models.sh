ResNet50_bs32_dygraph(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    ln -s ${data_path}/imagenet100_data/ ${cur_model_path}/dataset
    pip install -r requirements.txt
    rm -rf ./run_benchmark_dygraph.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/resnet/paddle/run_benchmark_resnet.sh ./run_benchmark_dygraph.sh
    sed -i '/set\ -xe/d' run_benchmark_dygraph.sh
    model_name=ResNet50_bs32_dygraph
    run_batchsize=32
    echo "индекс — скорость, 2GPU, начало, ResNet50_bs32_dygraph"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark_dygraph.sh 1 ${run_batchsize} ${model_name} mp 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_${model_name}_1_2_mp
}
ResNet50_pure_fp16_bs64(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    pip install -r requirements.txt
    pip install https://paddle-qa.bj.bcebos.com/benchmark/dali_temp_20230904/nvidia_dali_nightly_cuda110-1.30.0.dev20230901-9606845-py3-none-manylinux2014_x86_64.whl
    ln -s ${data_path}/imagenet100_data/ ${cur_model_path}/dataset
    rm -rf ./run_benchmark_dygraph.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/resnet/paddle/run_benchmark_resnet.sh ./run_benchmark_dygraph.sh
    sed -i '/set\ -xe/d' run_benchmark_dygraph.sh
    model_name=ResNet50_pure_fp16_bs64
    run_batchsize=64
    echo "индекс — скорость, 2GPU, начало, ResNet50_pure_fp16_bs64"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark_dygraph.sh 1 ${run_batchsize} ${model_name} mp 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_${model_name}_1_2_mp
}
ResNet50_bs32(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    pip install -r requirements.txt
    python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
    ln -s ${data_path}/imagenet100_data/ ${cur_model_path}/dataset
    rm -rf ./run_benchmark_static.sh
    cp ${BENCHMARK_ROOT}/static_graph/image_classification/paddle/run_benchmark_resnet.sh ./run_benchmark_static.sh
    sed -i '/set\ -xe/d' run_benchmark_static.sh
    model_name=ResNet50_bs32
    run_batchsize=32
    echo "индекс — скорость, 2GPU, начало, ResNet50_bs32_static"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark_static.sh 1 ${run_batchsize} ${model_name} mp 1 | tee ${BENCHMARK_ROOT}/logs/static/${model_name}_speed_2gpus 2>&1
    sleep 1s
    cat ${model_name}_1_2_mp
}
bert_base_seqlen128_fp32_bs32(){
    paddlenlp_dir=${BENCHMARK_ROOT}/PaddleNLP
    cd ${paddlenlp_dir}
    pip uninstall -y paddlenlp
    python setup.py install
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/model_zoo/bert/
    cd ${cur_model_path}
    ln -s ${data_path}/Bert/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en_seqlen512 ${cur_model_path}/wikicorpus_en_seqlen512
    mv wikicorpus_en_seqlen512 ./data
    ln -s ${data_path}/Bert/wikicorpus_en_seqlen128 ${cur_model_path}/wikicorpus_en_seqlen128
    mv wikicorpus_en_seqlen128 ./data
    rm -rf /root/.paddlenlp/models
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/bert/paddle/run_benchmark.sh ./run_benchmark.sh
    python -c 'import paddlenlp'
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_mode=base
    fp_mode=fp32
    seq_item=seqlen128
    bs_item=32
    model_name="bert_${model_mode}_${seq_item}_${fp_mode}_bs${bs_item}"
    echo "индекс — скорость, 2gpu, начало, ${model_name}"
    export FLAGS_call_stack_level=2
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} mp ${bs_item} 100  ${seq_item} | tee ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_${model_name}_1_2_mp
    unset FLAGS_call_stack_level
}
transformer_base_bs4096_amp_fp16(){
    pip install paddlenlp==2.0.5
    pip install attrdict
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/machine_translation/transformer
    cd ${cur_model_path}
    mkdir -p ~/.paddlenlp/datasets
    ln -s ${data_path}/transformer/WMT14ende ~/.paddlenlp/datasets/
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/transformer/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    mode_item=base
    fp_item=amp_fp16
    bs=4096
    model_name="Transformer_${mode_item}_bs${bs}_${fp_item}"
    echo "индекс — скорость, ${model_name} начало 2GPU"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh 1 mp 100 ${mode_item} ${fp_item} | tee  ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_${model_name}_1_2_mp
}
yolov3_bs8(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    git branch
    cd ${cur_model_path}
    pip install Cython
    pip install pycocotools
    pip install -r requirements.txt
    pip install numpy==1.20.3
    mkdir -p ~/.cache/paddle/weights
    ln -s ${prepare_path}/yolov3/DarkNet53_pretrained ~/.cache/paddle/weights
    cd ${cur_model_path}
    echo "-------перед подготовкой данных"
    rm -rf dataset/coco
    ln -s ${data_path}/COCO17 ./dataset/coco
    echo "-------после подготовки данных"
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/yolov3/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, 2GPU, начало"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh 1 mp 50 | tee ${BENCHMARK_ROOT}/logs/dynamic/yolov3_bs1_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_yolov3_1_2_mp
    echo "--------worklog.0"
    cat mylog/workerlog.0
    echo "--------worklog.1"
    cat mylog/workerlog.1
}
ppyolov2_bs6(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    git branch
    cd ${cur_model_path}
    pip install Cython
    pip install pycocotools
    pip install -r requirements.txt
    pip install numpy==1.20.3
    echo "-------перед подготовкой данных"
    rm -rf dataset/coco
    ln -s ${data_path}/COCO17 ./dataset/coco
    echo "-------после подготовки данных"
    rm -rf run_benchmark_ppyolov2.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/ppyolov2/paddle/run_benchmark_ppyolov2.sh ./
    sed -i '/set\ -xe/d' run_benchmark_ppyolov2.sh
    echo "индекс - скорость, 2gpu,ppyolov2 начать"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark_ppyolov2.sh 1 mp 50 | tee ${BENCHMARK_ROOT}/logs/dynamic/ppyolov2_bs6_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_ppyolov2_1_2_mp
    echo "--------worklog.0"
    cat ppyolov2_log/workerlog.0
    echo "--------worklog.1"
    cat ppyolov2_log/workerlog.1
}
TSM_bs16(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleVideo
    cd ${cur_model_path}
    pip install wget decord
    ln -s ${prepare_path}/tsm/ResNet50_pretrain.pdparams ${cur_model_path}/
    rm -rf ./data/ucf101
    ln -s ${data_path}/TSM/ucf101_Vedio/ ${cur_model_path}/data/ucf101
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/tsm/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 2GPU"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh  1 mp 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/tsm_bs_16_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_TSM_1_2_mp
}
deeplabv3_bs4_fp32(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleSeg/
    cd ${cur_model_path}
    pip install  visualdl
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/cityscapes_hrnet_torch ${cur_model_path}/data/cityscapes
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/seg_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    rm -rf ./output
    mkdir ./output
    model_item=deeplabv3
    bs_item=4
    echo "индекс — скорость, ${model_item} начало 2GPU"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh 1 ${bs_item} mp ${model_item} 50 | tee ${BENCHMARK_ROOT}/logs/dynamic/seg_${model_item}_bs${bs_item}_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_deeplabv3_bs4_fp32_1_2_mp
}
CycleGAN_bs1(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleGAN
    cd ${cur_model_path}
    if python -c "import pooch" >/dev/null 2>&1; then
        echo "Pooch уже установил, нужно удалить"
        pip uninstall -y pooch
    else
        echo "собачка не установлена"
    fi
    pip install -r requirements.txt
    pip install scikit-image==0.18.1
    pip install easydict
    mkdir -p data
    ln -s ${data_path}/cityscapes_gan_mini ${cur_model_path}/data/cityscapes
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/gan_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_item=CycleGAN
    echo "индекс — скорость, ${model_item} начало 1GPU"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp ${model_item} 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/gan_${model_item}_bs1_speed_1gpus 2>&1
    sleep 1s
    cat dynamic_CycleGAN_bs1_1_1_sp
}
mask_rcnn_bs1(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    cd ${cur_model_path}
    pip install Cython
    pip install pycocotools
    pip install -r requirements.txt
    package_check_list=(imageio tqdm Cython pycocotools tb_paddle scipy)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} уже установлено"
        else
            echo "${package} НЕ НАЙДЕН"
            pip install ${package}
            echo "${package} установлен"
        fi
    done
    ln -s ${prepare_path}/mask-rcnn/ResNet50_cos_pretrained  ~/.cache/paddle/weights
    cd ${cur_model_path}
    rm -rf dataset/coco
    ln -s ${data_path}/COCO17 ${cur_model_path}/dataset/coco
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/mask_rcnn/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 2GPU"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh  1 mp 100 | tee ${BENCHMARK_ROOT}/logs/dynamic/mask_rcnn_bs1_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_mask_rcnn_1_2_mp
}
PPOCR_mobile_2_bs8(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleOCR
    cd ${cur_model_path}
    pip install -r requirements.txt
    if python -c "import pooch" >/dev/null 2>&1; then
        echo "Pooch уже установил, нужно удалить"
        pip uninstall -y pooch
    else
        echo "собачка не установлена"
    fi
    package_check_list=(shapely scikit-image imgaug pyclipper lmdb tqdm numpy visualdl python-Levenshtein)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} уже установлено"
        else
            echo "${package} НЕ НАЙДЕН"
            pip install ${package}
            echo "${package} установлен"
        fi
    done
    rm -rf train_data/icdar2015
    if [ ! -d "train_data" ]; then
        mkdir train_data
    fi
    ln -s ${data_path}/PPOCR_mobile_2.0/icdar2015 ${cur_model_path}/train_data/icdar2015
    rm -rf pretrain_models
    ln -s ${prepare_path}/PPOCR_mobile_2.0/pretrain_models ./pretrain_models
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/ppocr_mobile_2/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 2GPU"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh  1 mp 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/ppocr_mobile_2_bs8_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_PPOCR_mobile_2_bs8_1_2_mp
}
seq2seq_bs128(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/seq2seq
    cd ${cur_model_path}
    ln -s ${data_path}/seq2seq/data/ ${cur_model_path}/data
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/seq2seq/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/seq2seq_bs128_speed_1gpus 2>&1
    sleep 1s
    cat dynamic_seq2seq_bs128_1_1_sp
}