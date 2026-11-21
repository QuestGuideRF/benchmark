#!/usr/bin/env bash
cur_model_list=(dy_bmn dy_tsn dy_tsm dy_slowfast dy_video_TimeSformer dy_bert dy_lac dy_transformer dy_yolov3 dy_gan dy_seg dy_seq2seq dy_resnet dy_ptb_medium dy_mobilenet dy_ppocr_mobile_2 dy_faster_rcnn_fpn \
dy_mask_rcnn_fpn dy_seg_repo dy_detection_repo dy_clas_repo dy_ocr_repo dy_gan_repo dy_speech_repo_pwgan dy_gpt dy_xlnet dy_speech_repo_conformer)
export log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
dy_seg_repo(){
    echo "dy_seg_repo"
    cur_model_path=${BENCHMARK_ROOT}/PaddleSeg/
    cd ${cur_model_path}
    sed -i '/set\ -xe/d' benchmark/run_benchmark.sh
    bash benchmark/run_all.sh
}
dy_gan_repo(){
    export FLAGS_cudnn_exhaustive_search=1
    echo "dy_gan_repo"
    cur_model_path=${BENCHMARK_ROOT}/PaddleGAN/
    cd ${cur_model_path}
    ln -s ${data_path}/dygraph_data/gan_repo/REDS/ ./data/
    pip install -v -e .
    sed -i '/set\ -xe/d' benchmark/run_benchmark.sh
    bash benchmark/run_all.sh
    unset FLAGS_cudnn_exhaustive_search
}
dy_speech_repo_pwgan(){
    echo "dy_speech_repo_pwgan"
    cur_model_path=${BENCHMARK_ROOT}/PaddleSpeech/
    cd ${cur_model_path}/tests/benchmark/pwgan/
    pip install -U setuptools==58.0.4
    pip install jsonlines
    apt-get install libsndfile1 -y
    pip install yacs -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install pytest-runner  -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install kaldiio  -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install loguru
    bash run_all.sh
}
dy_speech_repo_conformer(){
    echo " dy_speech_repo_conformer подготовить среду Python3 "
    cur_model_path=${BENCHMARK_ROOT}/PaddleSpeech/
    cd ${cur_model_path}/tests/benchmark/conformer/
    mkdir run_env
    ln -s $(which python3.7) run_env/python3
    ln -s $(which pip3.7) run_env/pip3
    export PATH=$(pwd)/run_env:${PATH}
    echo "dy_speech_repo_conformer"
    rm -rf ${cur_model_path}/dataset/aishell/aishell.py
    cp ${data_path}/dygraph_data/conformer/aishell.py ${cur_model_path}/dataset/aishell/
    apt-get install libsndfile1 -y
    pip install -U setuptools==58.0.4
    pip install yacs -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install pytest-runner  -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install kaldiio  -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install loguru
    echo "bash run.sh --stage 0 --stop_stage 0" >> prepare.sh
	bash prepare.sh
    pip install jsonlines
    pip list
    bash run.sh
    rm -rf ${BENCHMARK_ROOT}/PaddleSpeech/dataset/aishell
}
dy_video_TimeSformer(){
    export FLAGS_cudnn_exhaustive_search=1
    echo "dy_video_TimeSformer"
    cur_model_path=${BENCHMARK_ROOT}/PaddleVideo/
    cd ${cur_model_path}/
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
    pip install -r requirements.txt
    pip install scikit-image==0.18.2
    pip install pooch==1.5.2
    cd ${cur_model_path}/benchmark/TimeSformer/
    bash run_all.sh local
    rm -rf ${BENCHMARK_ROOT}/PaddleVideo/
    unset FLAGS_cudnn_exhaustive_search
}
dy_detection_repo(){
    echo "dy_detection_repo"
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection/
    pip install numpy -U
    cd ${cur_model_path}/
    sed -i '/set\ -xe/d' benchmark/run_benchmark.sh
    bash benchmark/run_all.sh
}
dy_ocr_repo(){
    echo "dy_ocr_repo"
    cur_model_path=${BENCHMARK_ROOT}/PaddleOCR/
    cd ${cur_model_path}/
    sed -i '/set\ -xe/d' benchmark/run_benchmark_det.sh
    bash benchmark/run_det.sh
}
dy_clas_repo(){
    echo "dy_clas_repo"
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas/
    cd ${cur_model_path}/
    package_check_list=(imageio tqdm Cython pycocotools tb_paddle scipy pandas wget h5py sklearn visualdl opencv-python)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} уже установлено"
        else
            echo "${package} НЕ НАЙДЕН"
            pip install ${package}
            echo "${package} установлен"
        fi
    done
    mkdir -p ./dataset/ILSVRC2012
    ln -s ${data_path}/dygraph_data/imagenet100_data/* ./dataset/ILSVRC2012
    sed -i '/set\ -xe/d' benchmark/run_benchmark.sh
    bash benchmark/run_all.sh
}
dy_bert(){
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
    cp ${BENCHMARK_ROOT}/dynamic_graph/bert/paddle/run_benchmark.sh ./run_benchmark.sh
    pip install paddlenlp
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_mode_list=(base large)
    fp_mode_list=(fp32 fp16)
    for model_mode in ${model_mode_list[@]}; do
        bs_list=(96)
        seq_item=(seqlen128)
        if [ ${model_mode} == "large" ]; then
            bs_list=(4)
            seq_item=(seqlen512)
        fi
        for fp_mode in ${fp_mode_list[@]}
        do
            for bs_item  in ${bs_list[@]}
            do
                model_name="bert_${model_mode}_${fp_mode}_${seq_item}_bs${bs_item}"
                echo "индекс – скорость, 1 ГБ, начало, ${model_name}"
                CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} sp ${bs_item} 500 ${seq_item} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
                sleep 60
                echo "индекс — скорость, 8 ГБ, run_mode — multi_process, начало, ${model_name}"
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} mp ${bs_item} 400  ${seq_item} | tee ${log_path}/${model_name}_speed_8gpus8p 2>&1
                sleep 60
            done
        done
    done
}
dy_mobilenet(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
    pip install -r requirements.txt
    ln -s ${data_path}/dygraph_data/imagenet100_data ${cur_model_path}/dataset/
    rm -f ./run_benchmark_mobilenet.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/mobilenet/paddle/run_benchmark_mobilenet.sh ./
    sed -i '/set\ -xe/d' run_benchmark_mobilenet.sh
    modle_list=(MobileNetV2 MobileNetV1)
    for model_name in ${modle_list[@]}
    do
        echo "------------> начать работу ${model_name}"
        echo "индекс — скорость, начало 1GPU"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark_mobilenet.sh 1  sp 1  ${model_name} | tee ${log_path}/dynamic_${model_name}_bs128_speed_1gpus 2>&1
        sleep 60
        echo "индекс - скорость, 8gpu, начало"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mobilenet.sh 1  mp  1 ${model_name} | tee ${log_path}/dynamic_${model_name}_bs128_speed_8gpus 2>&1
        sleep 60
    done
}
dy_seq2seq(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/seq2seq
    cd ${cur_model_path}
    ln -s ${data_path}/dygraph_data/seq2seq/data/ ${cur_model_path}/data
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/seq2seq/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 1 | tee ${log_path}/dynamic_seq2seq_bs128_speed_1gpus 2>&1
}
dy_ptb_medium(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/language_model/rnnlm
    cd ${cur_model_path}
    mkdir -p data
    ln -s ${data_path}/dygraph_data/ptb/simple-examples/ /root/.paddlenlp/datasets/PTB/
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/ptb/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 1 | tee ${log_path}/dynamic_ptb_medium_bs20_speed_1gpus 2>&1
}
dy_transformer(){
    echo "###########pip install Paddlenlp"
    cd ${BENCHMARK_ROOT}/PaddleNLP
    pip install -r requirements.txt
    pip install h5py
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
    mode_list=(big base)
    fp_list=(fp32 amp_fp16)
    for mode_item in ${mode_list[@]}
    do
        for fp_item in ${fp_list[@]}
        do
            model_name="Transformer_${mode_item}_${fp_item}"
            echo "индекс — скорость, ${model_name} начало 1GPU"
            CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp 600 ${mode_item} ${fp_item} | tee ${log_path}/dynamic_${model_name}_speed_1gpus 2>&1
            sleep 60
            echo "индекс — скорость, ${model_name} начало 8gpu, мп"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 500 ${mode_item} ${fp_item} | tee ${log_path}/dynamic_${model_name}_speed_8gpus 2>&1
        done
    done
}
dy_tsn(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleVideo
    cd ${cur_model_path}
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
    pip install -r requirements.txt
    pip install wget av
    pip install scikit-image==0.18.2
    pip install pooch==1.5.2
    rm -rf ResNet50_pretrain.pdparams
    ln -s ${prepare_path}/tsn/ResNet50_pretrain.pdparams ${cur_model_path}/
    rm -rf data
    ln -s ${data_path}/dygraph_data/TSN/data ${cur_model_path}/
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/tsn/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 32 TSN sp 1 | tee ${log_path}/dynamic_tsn_bs32_speed_1gpus 2>&1
    sleep 60
    echo "индекс - скорость, начало 8gpu, мп"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 32 TSN mp 1 | tee ${log_path}/dynamic_tsn_bs32_speed_8gpus 2>&1
}
dy_gan(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleGAN
    cd ${cur_model_path}
    if python -c "import pooch" >/dev/null 2>&1; then
        echo "Pooch уже установил, нужно удалить"
        pip uninstall -y pooch
    else
        echo "собачка не установлена"
    fi
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
    pip install -r requirements.txt
    pip install scikit-image==0.18.2
    mkdir -p data
    ln -s ${data_path}/dygraph_data/cityscapes_gan_mini ${cur_model_path}/data/cityscapes
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/gan_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_list=(Pix2pix CycleGAN)
    for model_item in ${model_list[@]}
    do
        echo "индекс — скорость, ${model_item} начало 1GPU"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp ${model_item} 1 | tee ${log_path}/dynamic_gan_${model_item}_bs1_speed_1gpus 2>&1
        sleep 10
    done
}
dy_seg(){
    export FLAGS_cudnn_exhaustive_search=1
    cur_model_path=${BENCHMARK_ROOT}/PaddleSeg/
    cd ${cur_model_path}
    pip install  visualdl
    pip install -U opencv-python==4.2.0.32
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/dygraph_data/cityscapes_hrnet_torch ${cur_model_path}/data/cityscapes
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/seg_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_list=(deeplabv3 HRnet)
    fp_mode_list=(fp32 fp16)
    for model_item in ${model_list[@]}
    do
        for fp_mode in ${fp_mode_list[@]}
        do
            if [ ${model_item} = "HRnet" ]; then
                bs_item=8
            elif [ ${model_item} = "deeplabv3" ]; then
                bs_item=4
            fi
            echo "индекс — скорость, ${model_item} начало 1GPU"
            CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 ${bs_item} sp ${model_item} 200 False ${fp_mode} | tee ${log_path}/dynamic_seg_${model_item}_bs${bs_item}_${fp_mode}_speed_1gpus 2>&1
            sleep 10
            echo "индекс — скорость, ${model_item} 8GPU, начало"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${bs_item} mp ${model_item} 200 False ${fp_mode} | tee ${log_path}/dynamic_seg_${model_item}_bs${bs_item}_${fp_mode}_speed_8gpus 2>&1
            sleep 10
        done
    done
    unset FLAGS_cudnn_exhaustive_search
}
dy_slowfast(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleVideo
    cd ${cur_model_path}
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
    pip install -r requirements.txt
    pip install tqdm
    pip install decord
    pip install pandas av
    pip install scikit-image==0.18.2
    pip install pooch==1.5.2
    echo "------------список пипсов"
    pip list
    rm -rf data
    ln -s ${data_path}/dygraph_data/slowfast/data/ ${cur_model_path}/
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/slowfast/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 sp 1 | tee ${log_path}/dynamic_slowfast_bs8_speed_1gpus 2>&1
    sleep 60
    echo "индекс - скорость, начало 8gpu, мп"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 1 | tee ${log_path}/dynamic_slowfast_bs8_speed_8gpus 2>&1
}
dy_mask_rcnn_fpn(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    cd ${cur_model_path}
    pip install Cython
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
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
    rm -rf dataset/coco/
    mkdir -p dataset/coco/
    ln -s ${data_path}/COCO17/annotations ${cur_model_path}/dataset/coco/annotations
    ln -s ${data_path}/COCO17/train2017 ${cur_model_path}/dataset/coco/train2017
    ln -s ${data_path}/COCO17/test2017 ${cur_model_path}/dataset/coco/test2017
    ln -s ${data_path}/COCO17/val2017 ${cur_model_path}/dataset/coco/val2017
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/mask_rcnn/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 sp 500 | tee ${log_path}/dynamic_mask_rcnn_fpn_bs4_speed_1gpus 2>&1
    sleep 60
    echo "индекс - скорость, начало 8gpu, мп"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 500 | tee ${log_path}/dynamic_mask_rcnn_fpn_bs4_speed_8gpus 2>&1
}
dy_yolov3(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    git branch
    cd ${cur_model_path}
    pip install Cython
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
    pip install -r requirements.txt
    pip install -U opencv-python==4.2.0.32
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
    echo "-------после подготовки данных"
    ls -l ./dataset/coco/
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/yolov3/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, 1 ГП, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 500 | tee ${log_path}/dynamic_yolov3_bs1_speed_1gpus 2>&1
    sleep 60
    echo "индекс — скорость, 8GPU, начало"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 500 | tee ${log_path}/dynamic_yolov3_bs1_speed_8gpus 2>&1
}
dy_tsm(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleVideo
    cd ${cur_model_path}
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
    pip install -r requirements.txt
    pip install wget av decord
    pip install scikit-image==0.18.2
    pip install pooch==1.5.2
    ln -s ${prepare_path}/tsm/ResNet50_pretrain.pdparams ${cur_model_path}/
    rm -rf ./data/ucf101
    ln -s ${data_path}/dygraph_data/TSM/ucf101_Vedio/ ${cur_model_path}/data/ucf101
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/tsm/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 sp 1 | tee ${log_path}/dynamic_tsm_bs_16_speed_1gpus 2>&1
    sleep 60
    echo "индекс - скорость, начало 8gpu, мп"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 1 | tee ${log_path}/dynamic_tsm_bs16_speed_8gpus 2>&1
}
dy_wavenet(){
    rm -rf ${BENCHMARK_ROOT}/SL_models/
    cp -r ${all_path}/SL_models/ ${BENCHMARK_ROOT}
    cur_model_path=${BENCHMARK_ROOT}/SL_models/benchmark/wavenet/paddle_implementation
    cd ${cur_model_path}
    ln -s ${data_path}/dygraph_data/wavenet/ljspeech ${cur_model_path}/
    apt-get install  libsndfile1 -y
    sed -i "s/opencv-python.*/opencv-python/g" ${data_path}/dygraph_data/wavenet/requirement.txt
    pip install -r ${data_path}/dygraph_data/wavenet/requirement.txt
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/wavenet/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    kill -9 `ps -ef|grep python |awk '{print $2}'`
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 sp | tee ${log_path}/dynamic_wavenet_bs8_speed_1gpus 2>&1
    sleep 60
    echo "индекс - скорость, начало 8gpu, мп"
    kill -9 `ps -ef|grep python |awk '{print $2}'`
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp | tee ${log_path}/dynamic_wavenet_bs8_speed_8gpus 2>&1
}
dy_senta(){
    rm -rf ${BENCHMARK_ROOT}/SL_models/
    cp -r ${all_path}/SL_models/ ${BENCHMARK_ROOT}
    cur_model_path=${BENCHMARK_ROOT}/SL_models/benchmark/senta/paddle2
    cd ${cur_model_path}
    ln -s ${data_path}/dygraph_data/senta/senta_data ${cur_model_path}/
    pip install tqdm
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/senta/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    net_list=(bow lstm bilstm gru bigru rnn birnn cnn)
    for net_item in ${net_list[@]}
    do
        echo "net — ${net_item}, индекс — скорость, начало 1GPU"
        model_name="Senta"_${net_item}_bs64
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${net_item} sp 2 | tee ${log_path}/dynamic_${model_name}_speed_1gpus 2>&1
        kill -9 `ps -ef|grep python |awk '{print $2}'`
        sleep 60
        echo "net — ${net_item}, индекс — скорость, начало 8GPU"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${net_item} mp 2 | tee ${log_path}/dynamic_${model_name}_speed_8gpus 2>&1
        kill -9 `ps -ef|grep python |awk '{print $2}'`
    done
}
dy_resnet(){
    export FLAGS_cudnn_exhaustive_search=1
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
    pip install -r requirements.txt
    ln -s ${data_path}/dygraph_data/imagenet100_data/ ${cur_model_path}/dataset
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/resnet/paddle/run_benchmark_resnet.sh ./
    sed -i '/set\ -xe/d' run_benchmark_resnet.sh
    model_list=(ResNet152_bs32 ResNet50_bs128 ResNet50_amp_fp16_bs128 ResNet50_amp_fp16_bs256 ResNet50_pure_fp16_bs128)
    for model_name in ${model_list[@]}
    do
	batch_size=${model_name
        echo "модель: ${model_name}, индекс — скорость, начало — 1 графический процессор"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_resnet.sh 1 ${batch_size} ${model_name} sp 1 | tee ${log_path}/dynamic_${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "модель: ${model_name}, индекс — скорость, начало 8GPU"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_resnet.sh 1 ${batch_size} ${model_name} mp 3 | tee ${log_path}/dynamic_${model_name}_speed_8gpus 2>&1
        sleep 60
    done
    unset FLAGS_cudnn_exhaustive_search
}
dy_lac(){
    cd ${BENCHMARK_ROOT}/PaddleNLP
    pip install -r requirements.txt
    pip uninstall -y paddlenlp
    pip install attrdict
    python setup.py install
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/lexical_analysis
    cd ${cur_model_path}
    ln -s ${data_path}/dygraph_data/lac/lexical_analysis_dataset_tiny/ ${cur_model_path}/data
    echo "###########pip install Paddlenlp"
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/lac/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 sp 10 | tee ${log_path}/dynamic_lac_bs32_speed_1gpus 2>&1
}
dy_ppocr_mobile_2() {
    cur_model_path=${BENCHMARK_ROOT}/PaddleOCR
    pip install fasttext
    cd ${cur_model_path}
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
    ln -s ${data_path}/dygraph_data/PPOCR_mobile_2.0/icdar2015 ${cur_model_path}/train_data/icdar2015
    rm -rf pretrain_models
    ln -s ${prepare_path}/PPOCR_mobile_2.0/pretrain_models ./pretrain_models
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/ppocr_mobile_2/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 sp 1 | tee ${log_path}/dynamic_ppocr_mobile_2_bs8_speed_1gpus 2>&1
    sleep 60
    echo "индекс - скорость, начало 8gpu, мп"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 1 | tee ${log_path}/dynamic_ppocr_mobile_2_bs8_speed_8gpus 2>&1
}
dy_bmn() {
    export FLAGS_cudnn_exhaustive_search=1
    cur_model_path=${BENCHMARK_ROOT}/PaddleVideo
    cd ${cur_model_path}
    pip install scikit-image==0.18.2
    pip install pooch==1.5.2
    package_check_list=(tqdm PyYAML numpy decord pandas av)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} уже установлено"
        else
            echo "${package} НЕ НАЙДЕН"
            pip install ${package}
            echo "${package} установлен"
        fi
    done
    rm -rf dataset
    ln -s ${data_path}/dygraph_data/bmn ${cur_model_path}/dataset
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/bmn/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 sp 1 | tee ${log_path}/dynamic_bmn_bs8_speed_1gpus 2>&1
    sleep 60
    echo "индекс - скорость, начало 8gpu, мп"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 1 | tee ${log_path}/dynamic_bmn_bs8_speed_8gpus 2>&1
    unset FLAGS_cudnn_exhaustive_search
}
dy_faster_rcnn_fpn() {
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    cd ${cur_model_path}
    pip install Cython
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
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
    rm -rf dataset/coco/
    mkdir -p dataset/coco/
    ln -s ${data_path}/COCO17/annotations ${cur_model_path}/dataset/coco/annotations
    ln -s ${data_path}/COCO17/train2017 ${cur_model_path}/dataset/coco/train2017
    ln -s ${data_path}/COCO17/test2017 ${cur_model_path}/dataset/coco/test2017
    ln -s ${data_path}/COCO17/val2017 ${cur_model_path}/dataset/coco/val2017
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/faster_rcnn_fpn/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало 1GPU"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 sp 500 | tee ${log_path}/dynamic_faster_rcnn_bs1_speed_1gpus 2>&1
    sleep 60
    echo "индекс - скорость, начало 8gpu, мп"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 500 | tee ${log_path}/dynamic_faster_rcnn_bs1_speed_8gpus 2>&1
}
dy_gpt(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP
    cd ${cur_model_path}
    pip uninstall paddlenlp -y
    sed -i "s/python3/python/g" examples/language_model/gpt/data_tools/Makefile
    sed -i "s/python-config/python3.7m-config/g" examples/language_model/gpt/data_tools/Makefile
    cat examples/language_model/gpt/data_tools/Makefile | grep python
    sed -i '/set\ -xe/d' tests/benchmark/run_benchmark.sh
    bash tests/benchmark/run_all.sh dygraph
}
dy_xlnet() {
    pip uninstall paddlenlp -y
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP
    cd ${cur_model_path}
    mkdir run_env
    run_env=$cur_model_path/run_env
    profile=${1:-"off"}
    rm -rf $run_env
    mkdir $run_env
    echo `which python3.7`
    ln -s $(which python3.7)m-config  $run_env/python3-config
    ln -s $(which python3.7) $run_env/python
    ln -s $(which pip3.7) $run_env/pip
    export PATH=$run_env:${PATH}
    sed -i "s/opencv-python.*/opencv-python/g" requirements.txt
    pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
    pip install sentencepiece -i https://mirror.baidu.com/pypi/simple
    pip install -e ./
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/xlnet/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_mode_list=(xlnet-base-cased)
    fp_item_list=(fp32)
    bs_item_list=(32 64 128)
    for model_mode in ${model_mode_list[@]}; do
        for fp_item in ${fp_item_list[@]}; do
            for bs_item in ${bs_item_list[@]}; do
                echo "индекс – скорость, 1 ГБ, начало, ${model_name}"
                run_mode=sp
                CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 300 ${model_mode} ${profile}  | tee ${log_path}/nlp_dygraph_xlnet_sp_bs${bs_item}_fp${fp_item}_speed_1gpus 2>&1
                echo "индекс — скорость, 8 ГБ, run_mode — multi_process, начало, ${model_name}"
                run_mode=mp
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 300 ${model_mode} ${profile}  | tee ${log_path}/nlp_dygraph_xlnet_mp_bs${bs_item}_fp${fp_item}_speed_8gpus 2>&1
                sleep 60
            done
        done
    done
}