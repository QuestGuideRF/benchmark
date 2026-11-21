#!/usr/bin/env bash
cur_model_list=(dy_to_static_bert dy_to_static_mobilenet dy_to_static_resnet dy_to_st_seg dy_to_st_transformer)
export log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
dy_to_static_bert() {
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/
    cd ${cur_model_path}
    pip install -r requirements.txt
    pip install h5py
    pip uninstall -y paddlenlp
    pip install attrdict
    python setup.py install
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/language_model/bert/
    cd ${cur_model_path}
    ln -s ${data_path}/Bert/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en_seqlen512 ${cur_model_path}/wikicorpus_en_seqlen512 ./data
    ln -s ${data_path}/Bert/wikicorpus_en_seqlen128 ./data
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_to_static/bert/paddle/run_benchmark.sh ./run_benchmark.sh
    pip install paddlenlp
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_mode_list=(base)
    fp_mode_list=(fp32 fp16)
    for model_mode in ${model_mode_list[@]}; do
        seq_list=(seqlen128)
        for fp_mode in ${fp_mode_list[@]}; do
            bs_list=(96)
            for bs_item in ${bs_list[@]}
            do
                for seq_item in ${seq_list[@]}
                do
                    model_name="bert_${model_mode}_${fp_mode}_${seq_item}_bs${bs_item}"
                    echo "индекс – скорость, 1 ГБ, начало, ${model_name}"
                    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} sp ${bs_item} 500 ${seq_item} | tee ${log_path}/dynamic_to_static_${model_name}_speed_1gpus 2>&1
                    sleep 60
                    echo "индекс — скорость, 8 ГБ, run_mode — multi_process, начало, ${model_name}"
                    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} mp ${bs_item} 400  ${seq_item} | tee ${log_path}/dynamic_to_static_${model_name}_speed_8gpus8p 2>&1
                    sleep 60
                done
            done
        done
    done
}
dy_to_static_mobilenet(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    pip install -r requirements.txt
    ln -s ${data_path}/dygraph_data/imagenet100_data ${cur_model_path}/dataset/
    rm -f ./run_benchmark_mobilenet.sh
    cp ${BENCHMARK_ROOT}/dynamic_to_static/mobilenet/paddle/run_benchmark_mobilenet.sh ./
    sed -i '/set\ -xe/d' run_benchmark_mobilenet.sh
    modle_list=(MobileNetV2 MobileNetV1)
    for model_name in ${modle_list[@]}
    do
        echo "------------> начать работу ${model_name}"
        echo "индекс — скорость, начало 1GPU"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark_mobilenet.sh 1  sp 1  ${model_name} | tee ${log_path}/dynamic_to_static_${model_name}_bs128_speed_1gpus 2>&1
        sleep 60
        echo "индекс - скорость, 8gpu, начало"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mobilenet.sh 1  mp  1 ${model_name} | tee ${log_path}/dynamic_to_static_${model_name}_bs128_speed_8gpus 2>&1
        sleep 60
    done
}
dy_to_static_yolov3(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    git branch
    cd ${cur_model_path}
    pip install Cython
    pip install -r requirements.txt
    if python -c "import pycocotools" >/dev/null 2>&1
    then
        echo "какаопи уже установлен"
    else
        echo "какаопи НЕ НАЙДЕН"
        cp -r ${prepare_path}/cocoapi/ ./
        cd cocoapi/PythonAPI/
        pip install Cython
        make install
        python setup.py install --user
        echo "Какаопи установлено"
    fi
    mkdir -p ~/.cache/paddle/weights
    ln -s ${prepare_path}/yolov3/DarkNet53_pretrained ~/.cache/paddle/weights
    cd ${cur_model_path}
    echo "-------перед подготовкой данных"
    ls -l ./dataset/coco/
    ln -s ${data_path}/coco/* ./dataset/coco/
    pip install -r ./dataset/coco/requirements.txt
    echo "-------после подготовки данных"
    ls -l ./dataset/coco/
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_to_static/yolov3/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, 1 ГП, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 500 | tee ${log_path}/dynamic_to_static_yolov3_speed_1gpus 2>&1
    sleep 60
    echo "индекс — скорость, 8GPU, начало"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 500 | tee ${log_path}/dynamic_to_static_yolov3_speed_8gpus 2>&1
}
dy_to_static_resnet(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    pip install -r requirements.txt
    ln -s ${data_path}/dygraph_data/imagenet100_data/ ${cur_model_path}/dataset
    rm -f ./run_benchmark_resnet.sh
    cp ${BENCHMARK_ROOT}/dynamic_to_static/resnet/paddle/run_benchmark_resnet.sh ./
    sed -i '/set\ -xe/d' run_benchmark_resnet.sh
    batch_size=32
    model_list=(ResNet50_bs128)
    for model_name in ${model_list[@]}
    do
        if [ ${model_name} == "ResNet50_bs128" ]; then
            batch_size=128
        fi
        echo "модель: ${model_name}, индекс — скорость, начало — 1 графический процессор"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_resnet.sh 1 ${batch_size} ${model_name} sp 1 | tee ${log_path}/dynamic_to_static_${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "модель: ${model_name}, индекс — скорость, начало 8GPU"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_resnet.sh 1 ${batch_size} ${model_name} mp 1 | tee ${log_path}/dynamic_to_static_${model_name}_speed_8gpus 2>&1
        sleep 60
    done
}
dy_to_st_seg(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleSeg/
    cd ${cur_model_path}
    pip install  visualdl scipy sklearn
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/dygraph_data/cityscapes_hrnet_torch ${cur_model_path}/data/cityscapes
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/seg_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_list=(deeplabv3 HRnet)
    for model_item in ${model_list[@]}
    do
        if [ ${model_item} = "HRnet" ]; then
            bs_item=8
        elif [ ${model_item} = "deeplabv3" ]; then
            bs_item=4
        fi
        echo "индекс — скорость, ${model_item} начало 1GPU"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 ${bs_item} sp ${model_item} 200 True fp32 | tee ${log_path}/dynamic_to_static_seg_${model_item}_bs${bs_item}_speed_1gpus 2>&1
        sleep 60
    done
}
dy_to_st_transformer(){
    echo "###########pip install Paddlenlp"
    cd ${BENCHMARK_ROOT}/PaddleNLP
    pip uninstall -y paddlenlp
    pip install attrdict
    python setup.py install
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/machine_translation/transformer
    cd ${cur_model_path}
    mkdir -p ~/.paddlenlp/datasets
    ln -s ${data_path}/dygraph_data/transformer/WMT14ende ~/.paddlenlp/datasets/
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/transformer/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    mode_list=(base)
    fp_list=(fp32 amp_fp16)
    for mode_item in ${mode_list[@]}
    do
        for fp_item in ${fp_list[@]}
        do
            model_name="Transformer_${mode_item}_${fp_item}"
            echo "индекс — скорость, ${model_name} начало 1GPU"
            CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp 600 ${mode_item} ${fp_item} True | tee ${log_path}/dynamic_to_static_${model_name}_speed_1gpus 2>&1
            sleep 60
            echo "индекс — скорость, ${model_name} начало 8gpu, мп"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 500 ${mode_item} ${fp_item} True | tee ${log_path}/dynamic_to_static_${model_name}_speed_8gpus 2>&1
        done
    done
}