#!/usr/bin/env bash
stage=0
stop_stage=100
echo "*******подготовить тест***********"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    export BENCHMARK_ROOT=/workspace
    log_date=`date "+%Y.%m%d.%H%M%S"`
    frame=pytorch1.9.0
    cuda_version=10.2
    save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/
    if [[ -d ${save_log_dir} ]]; then
        rm -rf ${save_log_dir}
    fi
    export TRAIN_LOG_DIR=${save_log_dir}/train_log
    mkdir -p ${TRAIN_LOG_DIR}
    log_path=${TRAIN_LOG_DIR}
fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    rm -rf run_env
    mkdir run_env
    ln -s $(which python3.7) run_env/python
    ln -s $(which pip3.7) run_env/pip
    export PATH=${BENCHMARK_ROOT}/run_env:${PATH}
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    pip install -U pip
    echo `pip --version`
    pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    pushd ./models/Parakeet/PWGAN/ParallelWaveGAN
    echo "--------------------------------$PWD"
    git reset --hard 8b7636b14b316ebb762c062abdb23645f6e45934
    pip install -e .
    popd
    git clone https://github.com/NVIDIA/apex.git
    pushd apex
    git checkout 3303b3e
    pip install -v --disable-pip-version-check --no-cache-dir ./
    pushd ../
    apt-get install nkf -y
    apt-get install sox -y
    apt-get install jq -y
fi
echo "*******Подготовка теста завершена***********"