#!/usr/bin/env bash
set -xe
echo "*******подготовить тест***********"
export BENCHMARK_ROOT=/workspace
run_env=$BENCHMARK_ROOT/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.8
cuda_version=10.2
save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/
if [[ -d ${save_log_dir} ]]; then
    rm -rf ${save_log_dir}
fi
export TRAIN_LOG_DIR=${save_log_dir}/train_log
mkdir -p ${TRAIN_LOG_DIR}
log_path=${TRAIN_LOG_DIR}
rm -rf $run_env
mkdir $run_env
echo `which python3.7`
ln -s $(which python3.7)m-config  $run_env/python3-config
ln -s $(which python3.7) $run_env/python
ln -s $(which pip3.7) $run_env/pip
export PATH=$run_env:${PATH}
echo `pip --version`
echo `python3-config --help`
pip install torch==1.8.0 -i https://mirror.baidu.com/pypi/simple
pip install regex pybind11 Ninja -i https://mirror.baidu.com/pypi/simple
cd $BENCHMARK_ROOT/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd -
echo "*******Подготовка теста завершена***********"