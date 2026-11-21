#!/usr/bin/env bash
set +ex
function LOG {
  echo [$0:${BASH_LINENO[0]}] $* >&2
}
LOG "[ИНФО] Запустить тестовый тест..."
BENCHMARK_ROOT=$(cd $(dirname $0)/../.. && pwd)
[ -z "$CUDA_VISIBLE_DEVICES" ] && CUDA_VISIBLE_DEVICES="0"
function prepare_env(){
  num_changed_files=$(git diff --name-only master | grep -E "api/(common)?(tests)?(_v2)?/(.*\.py|configs/.*\.json)" | wc -l)
  if [ ${num_changed_files} -eq 0 ]; then
    LOG "[ИНФО] Этот запрос на извлечение не меняет никаких файлов тестового теста, пропустите CI."
    exit 0
  fi
  LOG "[ИНФО] Идентификатор устройства: ${CUDA_VISIBLE_DEVICES}"
  LOG "[ИНФО] Обновить пип..."
  env http_proxy="" https_proxy="" pip install -U pip > /dev/null
  [ $? -ne 0 ] && LOG "[FATAL] Не удалось обновить пункт!" && exit -1
  PADDLE_WHL="paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"
  if [ ! -f "${PADDLE_WHL}" ]
  then
    PADDLE_WHL="paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl"
    PADDLE_URL="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/${PADDLE_WHL}"
    LOG "[ИНФО] Загрузка гребного колеса с ${PADDLE_URL} может занять несколько минут..."
    wget -q -O ${PADDLE_WHL} ${PADDLE_URL}
    [ $? -ne 0 ] && LOG "[FATAL] Не удалось загрузить гребное колесо!" && exit -1
  fi
  LOG "[ИНФО] Установка весла может занять несколько минут..."
  env http_proxy="" https_proxy="" pip install -U ${PADDLE_WHL} > /dev/null
  [ $? -ne 0 ] && LOG "[FATAL] Не удалось установить весло!" && exit -1
  for package in "факел==1.12.0" "torchvision" "тензорный поток == 2.3.0" "tensorflow-probability" "предварительная фиксация == 1,21" "пилинт==1.9.4" "pytest==4.6.9"
  do
    LOG "[ИНФО] Установка $package может занять несколько минут..."
    env http_proxy="" https_proxy="" pip install $package -i https://pypi.tuna.tsinghua.edu.cn/simple > /dev/null
    [ $? -ne 0 ] && LOG "[FATAL] Не удалось установить $package!" && exit -1
  done
  python -c "import tensorflow as tf; print(tf.__version__)" > /dev/null
  [ $? -ne 0 ] && LOG "[FATAL] Установка tensorflow прошла успешно, но он не работает!" && exit -1
  python -c "import torch ; print(torch.__version__)" > /dev/null
  [ $? -ne 0 ] && LOG "[FATAL] Установите pytorch успешно, но он не работает!" && exit -1
  apt-get update > /dev/null 2> /dev/null
}
function run_api(){
  LOG "[ИНФО] Начать запуск теста API..."
  API_NAMES=()
  for file in $(git diff --name-only master | grep -E "api/tests(_v2)?/(.*\.py|configs/.*\.json)")
  do
    LOG "[INFO] Обнаружено изменение ${file}."
    api=${file
    [[ "$api" =~ "common_import" ]] && continue
    [[ "$api" =~ "test_main" ]] && continue
    [[ "$api" =~ "__init__" ]] && continue
    [ -f "${BENCHMARK_ROOT}/api/${api}.py" ] && API_NAMES[${
    if [[ "$файл" =~ ".json" ]]
    then
      [ -f "${BENCHMARK_ROOT}/api/${api/configs\//}.py" ] && API_NAMES[${
      for sub_file in $(grep -l "APIConfig(.${api##*/}.)" ${BENCHMARK_ROOT}/api/tests_v2/*.py)
      do
        sub_api=${sub_file
      done
    fi
  done
  API_NAMES=($(echo ${API_NAMES[@]} | tr ' ' '\n' | sort | uniq))
  [ -z "$(echo ${API_NAMES[@]} | grep -w 'тесты')" ] && API_NAMES[${
  [ -z "$(echo ${API_NAMES[@]} | grep -w 'test_v2')" ] && API_NAMES[${
  LOG "[ИНФО] Будут работать следующие API: ${API_NAMES[@]}"
  fail_name=()
  for name in ${API_NAMES[@]}
  do
    for device_type in "GPU"
    do
      [ $device_type == "GPU" ] && device_limit="" || device_limit="окружение CUDA_VISIBLE_DEVICES="
      ${device_limit} bash ${BENCHMARK_ROOT}/api/${name%/*}/run.sh ${name
      [ $? -ne 0 ] && fail_name[${
    done
  done
  if [ ${
  then
    LOG "[FATAL] Неудачные тесты API: ${fail_name[@]}"
    echo ${fail_name[@]}
    exit -1
  fi
}
function check_style(){
  LOG "[ИНФО] Начать проверку стиля кода..."
  pre-commit uninstall >&2
  pre-commit install >&2
  commit_files=on
  LOG "[ИНФО] Проверьте стиль кода с помощью каждой фиксации, это может занять несколько минут..."
  filenames=$(git diff --name-only upstream/master)
  env http_proxy="" https_proxy="" pre-commit run --files $filenames >&2 || commit_files=off
  [ $commit_files == 'off' ] && git diff && return -1 || return 0
}
function summary_problems(){
  local check_style_code=$1
  local check_style_info=$2
  local run_api_code=$3
  local run_api_info=$4
  if [ $check_style_code -ne 0 -o $run_api_code -ne 0 ]
  then
    LOG "[FATAL] ============================================"
    LOG "[FATAL] Общие проблемы:"
    if [ $check_style_code -ne 0 -a $run_api_code -ne 0 ]
    then
      LOG "[FATAL] Есть 2 ошибки: ошибка стиля кода и ошибка теста API."
    else
      [ $check_style_code -ne 0 ] && LOG "[FATAL] Имеется 1 ошибка: ошибка стиля кода."
      [ $run_api_code -ne 0 ] && LOG "[FATAL] Возникла 1 ошибка: ошибка теста API."
    fi
    LOG "[FATAL] ============================================"
    if [ $check_style_code -ne 0 ]
    then
      LOG "[FATAL] === Ошибка стиля кода. Исправьте ее в соответствии с информацией о различиях:"
      echo "$check_style_info"
    fi
    if [ $run_api_code -ne 0 ]
    then
      LOG "[FATAL] === Ошибка теста API. Исправьте неудачные тесты API в соответствии с фатальным журналом:"
      LOG "[FATAL] $run_api_info"
    fi
    [ $check_style_code -ne 0 ] && exit $check_style_code
    [ $run_api_code -ne 0 ] && exit $run_api_code
  fi
}
function main(){
  prepare_env
  check_style_info=$(check_style)
  check_style_code=$?
  [ -z "${check_style_info}" ] && check_style_code=0
  run_api_info=$(run_api)
  run_api_code=$?
  summary_problems $check_style_code "$check_style_info" $run_api_code "$run_api_info"
  LOG "[ИНФО] Тестовый тест выполнен успешно, ошибок нет!"
}
main