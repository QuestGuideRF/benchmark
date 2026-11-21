#!/usr/bin/env bash
echo "*******подготовить тест***********"
export BENCHMARK_ROOT=/workspace
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.0.0
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
git clone https://github.com/lzzyzlbb/first-order-model
git checkout add_log
cd first-order-model
pip install -r requirements.txt
imageio_download_bin ffmpeg
mkdir -p data
wget https://paddlegan.bj.bcebos.com/datasets/fom_test_data.tar \
    -O data/fom_test_data.tar
tar -vxf data/fom_test_data.tar -C data/
mv data/first_order/Voxceleb/ data/vox-png/
echo "*******Подготовка теста завершена***********"