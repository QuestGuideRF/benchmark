#!/usr/bin/env bash
echo "*******подготовить тест***********"
export BENCHMARK_ROOT=/workspace
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.10.0
cuda_version=10.2
save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/
if [[ -d ${save_log_dir} ]]; then
    rm -rf ${save_log_dir}
fi
export TRAIN_LOG_DIR=${save_log_dir}/train_log
mkdir -p ${TRAIN_LOG_DIR}
log_path=${TRAIN_LOG_DIR}
rm -rf run_env
mkdir run_env
ln -s $(which python3.7) run_env/python
ln -s $(which pip3.7) run_env/pip
export PATH=/workspace/run_env:${PATH}
pip install -U pip
echo `pip --version`
pip install torch==1.10.0 -i https://mirror.baidu.com/pypi/simple
pip install git+https://github.com/huggingface/transformers
pip install accelerate -i https://mirror.baidu.com/pypi/simple
pip install datasets>=1.8.0 -i https://mirror.baidu.com/pypi/simple
pip install sentencepiece!=0.1.92 -i https://mirror.baidu.com/pypi/simple
pip install scipy -i https://mirror.baidu.com/pypi/simple
pip install scikit-learn -i https://mirror.baidu.com/pypi/simple
pip install protobuf -i https://mirror.baidu.com/pypi/simple
echo "*******Подготовка теста завершена***********"