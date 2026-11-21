#!/usr/bin/env bash
cur_model_list=(image_classification seg_model transformer bert gpt)
export log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
seg_model(){
    cd ${BENCHMARK_ROOT}/PaddleSeg/
    git checkout origin/benchmark
    git branch
    cur_model_path=${BENCHMARK_ROOT}/PaddleSeg/legacy
    cd ${cur_model_path}
    ln -s ${data_path}/cityscape ${cur_model_path}/dataset/cityscapes
    ln -s ${prepare_path}/deeplabv3p_xception65_bn_cityscapes ${cur_model_path}/pretrained_model/
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/seg_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, 1 ГП, начало"
    model_list=(deeplabv3 HRnet)
    bs_item=2
    for model_item in ${model_list[@]}
    do
        if [ ${model_item} = "HRnet" ]; then
            bs_item=8
        fi
        echo "индекс — скорость, ${model_item} начало 1GPU"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp ${model_item} 180 ${bs_item} | tee ${log_path}/${FUNCNAME}_${model_item}_speed_1gpus 2>&1
        sleep 60
        echo "индекс — скорость, ${model_item} 8GPU, начало"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp ${model_item} 180 ${bs_item} | tee ${log_path}/${FUNCNAME}_${model_item}_speed_8gpus8p 2>&1
        sleep 60
    done
}
image_classification(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    pip install -r requirements.txt
    ln -s ${data_path}/dygraph_data/imagenet100_data/ ${cur_model_path}/dataset/
    ln -s ${data_path}/ILSVRC2012 ${cur_model_path}/dataset/
    rm -rf ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/image_classification/paddle/run_benchmark_resnet.sh ./run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_list=(SE_ResNeXt50_32x4d_bs32 ResNet101_bs32 ResNet50_bs128 ResNet50_bs256)
    for model_name in ${model_list[@]}; do
        run_batchsize=${model_name
        echo "индекс – скорость, 1 графический процессор, начало, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${run_batchsize} ${model_name} sp 1 | tee ${log_path}/${FUNCNAME}_${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "индекс — скорость, 8 ГБ, run_mode — multi_process, начало, ${model_name}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${run_batchsize} ${model_name} mp 3 | tee ${log_path}/${FUNCNAME}_${model_name}_speed_8gpus8p 2>&1
        sleep 60
    done
    rm -rf run_benchmark_fp16.sh
    cp ${BENCHMARK_ROOT}/static_graph/image_classification/resnet50_fp/paddle/run_benchmark_fp16.sh ./run_benchmark_fp16.sh
    sed -i '/set\ -xe/d' run_benchmark_fp16.sh
    mode_list=(amp pure)
    for fp_mode in ${mode_list[@]}
    do
        bs_list=(128)
        for bs_item in ${bs_list[@]}
        do
            model_name="ResNet50_bs${bs_item}_${fp_mode}_fp16"
            echo "bs=${bs_item}, модель: ${model_name}, индекс – скорость, 1 графический процессор, run_mode – sp, начало"
            CUDA_VISIBLE_DEVICES=0 bash run_benchmark_fp16.sh 1 ${bs_item} sp 1 ${fp_mode} | tee ${log_path}/${FUNCNAME}_${model_name}_speed_1gpus 2>&1
            if [ ${bs_item} == 128 ]; then
                sleep 60
                echo "bs=${bs_item}, модель – ${model_name}, индекс – скорость, 8 графических процессоров, режим выполнения – multi_process, начало"
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_fp16.sh 1 ${bs_item} mp 2 ${fp_mode}  | tee ${log_path}/${FUNCNAME}_${model_name}_speed_8gpus8p 2>&1
            fi
            sleep 60
        done
    done
}
detection(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection/static
    cd ${cur_model_path}
    git branch
    ls tools
    rm -rf dataset/coco/
    mkdir -p dataset/coco/
    ln -s ${data_path}/COCO17/annotations ${cur_model_path}/dataset/coco/annotations
    ln -s ${data_path}/COCO17/train2017 ${cur_model_path}/dataset/coco/train2017
    ln -s ${data_path}/COCO17/test2017 ${cur_model_path}/dataset/coco/test2017
    ln -s ${data_path}/COCO17/val2017 ${cur_model_path}/dataset/coco/val2017
    mkdir -p ~/.cache/paddle/weights
    ln -s ${prepare_path}/detection/ResNet101_vd_pretrained ~/.cache/paddle/weights
    ln -s ${prepare_path}/detection/ResNet50_cos_pretrained ~/.cache/paddle/weights
    ln -s ${prepare_path}/detection/ResNeXt101_vd_64x4d_pretrained ~/.cache/paddle/weights
    package_check_list=(imageio tqdm Cython pycocotools tb_paddle scipy pandas wget h5py)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} уже установлено"
        else
            echo "${package} НЕ НАЙДЕН"
            pip install ${package}
            echo "${package} установлен"
        fi
    done
    cp ${BENCHMARK_ROOT}/static_graph/Detection/paddle/run_benchmark.sh ./run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh
    batch_size=2
    model_list=(mask_rcnn_fpn_resnet mask_rcnn_fpn_resnext cascade_rcnn_fpn)
    for model_name in ${model_list[@]}; do
        echo "индекс – скорость, 1 графический процессор, начало, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_name} sp 600 | tee ${log_path}/${FUNCNAME}_${model_name}_bs${batch_size}_speed_1gpus 2>&1
        sleep 60
        echo "индекс – скорость, 1 графический процессор, профилировщик включен, ${model_name}"
    done
}
mask_rcnn(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection/static
    cd ${cur_model_path}
    git branch
    if python -c "import pycocotools" >/dev/null 2>&1
    then
        echo "какаопи уже установлен"
    else
        echo "какаопи НЕ НАЙДЕН"
        cp -r ${prepare_path}/cocoapi/ ./
        cd cocoapi/PythonAPI/
        pip install Cython
        make install
        python2 setup.py install --user
        echo "Какаопи установлено"
    fi
    if python -c "import tb_paddle" >/dev/null 2>&1;
    then
        echo "tb_paddle уже установлен"
    else
        echo "tb_paddle НЕ НАЙДЕН"
        pip install tb_paddle
        echo "tb_paddle установлен"
    fi
    mkdir -p /root/.cache/paddle/weights
    ln -s ${prepare_path}/mask-rcnn/ResNet50_cos_pretrained  ~/.cache/paddle/weights
    cd ${cur_model_path}
    rm -rf dataset/coco/
    mkdir -p dataset/coco/
    ln -s ${data_path}/COCO17/annotations ${cur_model_path}/dataset/coco/annotations
    ln -s ${data_path}/COCO17/train2017 ${cur_model_path}/dataset/coco/train2017
    ln -s ${data_path}/COCO17/test2017 ${cur_model_path}/dataset/coco/test2017
    ln -s ${data_path}/COCO17/val2017 ${cur_model_path}/dataset/coco/val2017
    cp ${BENCHMARK_ROOT}/static_graph/Mask-RCNN/paddle/run_benchmark.sh ./run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, 1 ГП, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 600 | tee ${log_path}/${FUNCNAME}_bs1_speed_1gpus 2>&1
    sleep 60
    echo "индекс — скорость, 1 графический процессор, профилировщик включен, начало"
    echo "индекс — maxbs, 1gpu, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 6 sp 200 | tee ${log_path}/${FUNCNAME}_bs1_maxbs_1gpus 2>&1
    sleep 60
    echo "индекс — скорость, 8 ГБ, run_mode — multi_process, начало"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 600 | tee ${log_path}/${FUNCNAME}_bs1_speed_8gpus8p 2>&1
}
bert(){
    cd ${BENCHMARK_ROOT}/PaddleNLP
    pip install -r requirements.txt
    pip install h5py
    pip uninstall -y paddlenlp
    pip install attrdict
    python setup.py install
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/language_model/bert/static
    cd ${cur_model_path}
    ln -s ${data_path}/Bert/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en_seqlen512 ${cur_model_path}/wikicorpus_en_seqlen512
    ln -s ${data_path}/Bert/wikicorpus_en_seqlen128 ${cur_model_path}/wikicorpus_en_seqlen128
    cp ${BENCHMARK_ROOT}/static_graph/BERT/paddle/run_benchmark.sh ./run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_mode_list=(base large)
    fp_mode_list=(fp32 fp16)
    for model_mode in ${model_mode_list[@]}; do
        seq_list=(seqlen128)
        if [ ${model_mode} == "large" ]; then
            seq_list=(seqlen512)
            bs_list=(10)
        fi
        for fp_mode in ${fp_mode_list[@]}; do
            if [ ${model_mode} == "base" ] && [ ${fp_mode} == "fp32" ]; then
                bs_list=(32)
            elif [ ${model_mode} == "base" ] && [ ${fp_mode} == "fp16" ]; then
                bs_list=(96)
            fi
            for bs_item in ${bs_list[@]}
            do
                for seq_item in ${seq_list[@]}
                do
                    model_name="${FUNCNAME}_${model_mode}_${fp_mode}_${seq_item}_bs${bs_item}"
                    echo "индекс – скорость, 1 ГБ, начало, ${model_name}"
                    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} sp ${bs_item} 500 ${seq_item} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
                    sleep 60
                    echo "индекс — скорость, 8 ГБ, run_mode — multi_process, начало, ${model_name}"
                    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} mp ${bs_item} 400  ${seq_item} | tee ${log_path}/${model_name}_speed_8gpus8p 2>&1
                    sleep 60
                done
            done
        done
    done
}
transformer(){
    cd ${BENCHMARK_ROOT}/PaddleNLP
    pip install -r requirements.txt
    pip install h5py
    pip uninstall -y paddlenlp
    pip install attrdict
    python setup.py install
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/machine_translation/transformer/static
    cd ${cur_model_path}
    mkdir -p ~/.paddlenlp/datasets
    ln -s ${data_path}/dygraph_data/transformer/WMT14ende ~/.paddlenlp/datasets/
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/Transformer/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    mode_list=(big base)
    fp_list=(fp32 amp_fp16 pure_fp16)
    for mode_item in ${mode_list[@]}
    do
        for fp_item in ${fp_list[@]}
        do
            if [ ${mode_item} == "big" ] && [ ${fp_item} == "fp32" ]; then
                bs_list=(2560)
            elif [ ${mode_item} == "big" ] && [ ${fp_item} == "pure_fp16" ]; then
                bs_list=(5120)
            else
               bs_list=(4096)
            fi
            for bs_item in ${bs_list[@]}
            do
                model_name="Transformer_${mode_item}_bs${bs_item}_${fp_item}"
                echo "индекс — скорость, ${model_name} начало 1GPU"
                CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp 600 ${mode_item} ${fp_item} ${bs_item} | tee   ${log_path}/${model_name}_speed_1gpus 2>&1
                sleep 60
                echo "индекс — скорость, ${model_name} начало 8gpu, мп"
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 500 ${mode_item} ${fp_item} ${bs_item} | tee  ${log_path}/${model_name}_speed_8gpus8p 2>&1
            done
        done
    done
}
yolov3(){
    if python -c "import pycocotools" >/dev/null 2>&1
    then
        echo "какаопи уже установлен"
    else
        echo "какаопи НЕ НАЙДЕН"
        cp -r ${prepare_path}/cocoapi/ ./
        cd cocoapi/PythonAPI/
        pip install Cython
        make install
        python2 setup.py install --user
        echo "Какаопи установлено"
    fi
    cd ${BENCHMARK_ROOT}/PaddleDetection/static
    git branch
    mkdir -p ~/.cache/paddle/weights
    ln -s ${prepare_path}/yolov3/DarkNet53_pretrained ~/.cache/paddle/weights
    rm -rf dataset/coco
    ln -s ${data_path}/coco ./dataset/coco
    cp ${BENCHMARK_ROOT}/static_graph/yolov3/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, 1 ГП, начало"
    bs_item=16
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 600 | tee ${log_path}/${FUNCNAME}_bs${bs_item}_speed_1gpus 2>&1
    sleep 60
    echo "индекс — maxbs, 1gpus, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 6 sp 600 | tee ${log_path}/${FUNCNAME}_maxbs_1gpus 2>&1
    sleep 60
    echo "индекс — скорость, 8 ГБ, run_mode — multi_process, начало"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 600 | tee ${log_path}/${FUNCNAME}_bs${bs_item}_speed_8gpus8p 2>&1
}
ddpg_deep_explore(){
    cur_model_path=${BENCHMARK_ROOT}/DDPG_Deep_Explore/Fluid_version
    cd ${cur_model_path}
    if python -c "import parl" >/dev/null 2>&1
    then
        echo "парл уже установлен"
    else
        echo "парл НЕ НАЙДЕН"
        pip install parl==1.1
        echo "Парл установлен"
    fi
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
}
paddingrnn(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/legacy/language_model
    cd ${cur_model_path}
    ln -s ${data_path}/simple-examples ${cur_model_path}/data/simple-examples
    cp ${BENCHMARK_ROOT}/static_graph/PaddingRNN/lstm_paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_type_list=(small large)
    rnn_type_list=(static padding)
    for model_type in ${model_type_list[@]}; do
        for rnn_type in ${rnn_type_list[@]}; do
        model_name="${FUNCNAME}_${model_type}_${rnn_type}_bs20"
        echo "индекс — скорость, 1gpu, ${model_name}, начало"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_type} ${rnn_type} sp 3 | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "индекс – скорость, 1 графический процессор, ${model_name}, профилировщик включен, начните"
        sleep 60
        done
    done
}
seq2seq(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/legacy/seq2seq/seq2seq/
    cd ${cur_model_path}
    rm -r ${cur_model_path}/data
    mkdir ${cur_model_path}/data
    ln -s ${data_path}/seq2seq_paddle/en-vi ${cur_model_path}/data
    cp ${BENCHMARK_ROOT}/static_graph/seq2seq/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 1 | tee ${log_path}/${FUNCNAME}_bs128_speed_1gpus 2>&1
}
CycleGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/horse2zebra/ ${cur_model_path}/data/cityscapes
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/CycleGAN/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 600 | tee ${log_path}/${FUNCNAME}_bs1_speed_1gpus 2>&1
}
StarGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}
    rm -r ${cur_model_path}/data
    mkdir -p ${cur_model_path}/data/celeba
    ln -s ${data_path}/CelebA/Anno/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Eval/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Img/img_align_celeba ${cur_model_path}/data/celeba/
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio уже установлено"
    else
        echo "imageio НЕ НАЙДЕНО"
        pip install imageio
        echo "imageio установлено"
    fi
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/StarGAN/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 300 | tee ${log_path}/${FUNCNAME}_bs16_speed_1gpus 2>&1
    sleep 60
    echo "индекс — скорость, профилировщик включен, начинаем"
}
AttGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}
    rm -r ${cur_model_path}/data
    mkdir -p ${cur_model_path}/data/celeba
    ln -s ${data_path}/CelebA/Anno/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Eval/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Img/img_align_celeba ${cur_model_path}/data/celeba/
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio уже установлено"
    else
        echo "imageio НЕ НАЙДЕНО"
        pip install imageio
        echo "imageio установлено"
    fi
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/AttGAN/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
}
STGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}
    rm -r ${cur_model_path}/data
    mkdir -p ${cur_model_path}/data/celeba
    ln -s ${data_path}/CelebA/Anno/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Eval/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Img/img_align_celeba ${cur_model_path}/data/celeba/
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio уже установлено"
    else
        echo "imageio НЕ НАЙДЕНО"
        pip install imageio
        echo "imageio установлено"
    fi
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/STGAN/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 300 | tee ${log_path}/${FUNCNAME}_bs32_speed_1gpus 2>&1
    sleep 60
    echo "индекс — скорость, профилировщик включен, начинаем"
}
CGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}
    rm -r ${cur_model_path}/data
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/mnist ${cur_model_path}/data
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio уже установлено"
    else
        echo "imageio НЕ НАЙДЕНО"
        pip install imageio
        echo "imageio установлено"
    fi
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/GAN_models/PaddleGAN/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_list=(CGAN DCGAN)
    for model_name in ${model_list[@]}; do
        echo "индекс — скорость, начало, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 CGAN sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
    done
}
Pix2pix(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}
    rm -r ${cur_model_path}/data
    mkdir ${cur_model_path}/data
    ln -s ${data_path}/Pix2pix/cityscapes ${cur_model_path}/data
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio уже установлено"
    else
        echo "imageio НЕ НАЙДЕНО"
        pip install imageio
        echo "imageio установлено"
    fi
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/GAN_models/PaddleGAN/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "индекс — скорость, начало"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 Pix2pix sp 600 | tee ${log_path}/${FUNCNAME}_bs1_speed_1gpus 2>&1
    sleep 60
    echo "индекс — скорость, профилировщик включен, начинаем"
}
nextvlad(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/video/
    cd ${cur_model_path}
    rm -rf data
    mkdir -p data/dataset
    ln -s ${data_path}/youtube8m_paddle ./data/dataset/youtube8m
    ln -s ${data_path}/ctcn_paddle/ ./data/dataset/ctcn
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/train/* > ./data/dataset/youtube8m/train.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/val/* > ./data/dataset/youtube8m/val.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/val/* > ./data/dataset/youtube8m/test.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/val/* > ./data/dataset/youtube8m/infer.list
    package_check_list=(imageio tqdm Cython pycocotools pandas wget h5py)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} уже установлено"
        else
            echo "${package} НЕ НАЙДЕН"
            pip install ${package}
            echo "${package} установлен"
        fi
    done
    cp ${BENCHMARK_ROOT}/static_graph/NextVlad/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_list=(nextvlad)
    for model_name in ${model_list[@]}; do
        echo "индекс – скорость, 1 графический процессор, начало, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 32 ${model_name} sp 2 | tee ${log_path}/${model_name}_bs32_speed_1gpus 2>&1
        sleep 60
        echo "индекс — скорость, 1 графический процессор, prfoiler включен, начало, ${model_name}"
    done
}
gpt(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP
    cd ${cur_model_path}
    pip uninstall paddlenlp -y
    echo "среда Python - это"
    ls -l ${BENCHMARK_ROOT}/run_env/
    sed -i "s/python3/python/g" examples/language_model/gpt/data_tools/Makefile
    sed -i "s/python-config/python3.7m-config/g" examples/language_model/gpt/data_tools/Makefile
    cat examples/language_model/gpt/data_tools/Makefile | grep python
    sed -i '/set\ -xe/d' tests/benchmark/run_benchmark.sh
    bash tests/benchmark/run_all.sh static
}