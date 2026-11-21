#!/usr/bin/env bash
echo "*******подготовить тест***********"
export BENCHMARK_ROOT=/workspace
log_path=${TRAIN_LOG_DIR:-"pwd"}
pip3.7 install -U pip -i https://mirror.baidu.com/pypi/simple
echo `pip --version`
pip3.7 install pqi -i https://mirror.baidu.com/pypi/simple
pqi add baidu https://mirror.baidu.com/pypi/simple
pqi use baidu
pip3.7 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3.7 install anyconfig
echo "*******Подготовка теста завершена***********"