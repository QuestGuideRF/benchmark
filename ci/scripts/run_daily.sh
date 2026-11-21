#!/usr/bin/env bash
function LOG {
  echo "[$(базовое имя $0):${BASH_LINENO[0]}]" $* >&2
}
function build_paddle() {
  [ -d Paddle ] || git clone https://github.com/PaddlePaddle/Paddle.git
  pushd Paddle > /dev/null || exit
  git pull
  LOG "[ИНФО] Идентификатор фиксации изображения: $(git log | head -n1 | awk '{print $2}')"
  version=$(date -d @$(git log -1 --pretty=format:%ct) "+%Y.%m%d.%H%M%S")
  branch="develop"
  PADDLE_VERSION=${version}.${branch}.gcc82
  IMAGE_NAME=paddlepaddle_gpu-0.0.0.${PADDLE_VERSION}-cp37-cp37m-linux_x86_64.whl
  LOG "[ИНФО] ИМЯ_ИЗОБРАЖЕНИЯ: ${IMAGE_NAME}"
  [ -f build/python/dist/${IMAGE_NAME} ] && popd >/dev/null && return 0
  env CMAKE_BUILD_TYPE=Release                \
      PY_VERSION=3.7                          \
      PADDLE_VERSION=0.0.0.${PADDLE_VERSION}  \
      WITH_AVX=ON                             \
      WITH_GPU=ON                             \
      WITH_TEST=OFF                           \
      WITH_TESTING=OFF                        \
      RUN_TEST=OFF                            \
      WITH_PYTHON=ON                          \
      WITH_STYLE_CHECK=OFF                    \
      CMAKE_EXPORT_COMPILE_COMMANDS=ON        \
      WITH_MKL=ON                             \
      BUILD_TYPE=Release                      \
      WITH_DISTRIBUTE=OFF                     \
      CMAKE_VERBOSE_MAKEFILE=OFF              \
      /bin/bash paddle/scripts/paddle_build.sh build $(nproc)
  if [ $? -eq 0 -a -f build/python/dist/${IMAGE_NAME} ]
  then
    LOG "[ИНФО] Успех в сборке Paddle!"
  else
    LOG "[ИНФО] Не удалось построить весло!"
    exit -1
  fi
  popd > /dev/null
  mv Paddle/build/python/dist/${IMAGE_NAME} paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
}
function run(){
  [ -f paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl ] || exit -1
  mkdir -p logs
  logs_dir=$(pwd)/logs
  for package in "paddlepaddle" "paddlepaddle-gpu"
  do
    LOG "[ИНФО] Удалить $package..."
    pip uninstall -y $package > /dev/null
  done
  for package in "nvidia-ml-py" "тензорный поток == 2.3.0" "paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"
  do
    LOG "[ИНФО] Установить $package..."
    pip install $package > /dev/null
  done
  pushd api > /dev/null
  bash deploy/main_control.sh tests_v2 tests_v2/configs ${logs_dir} "${VISIBLE_DEVICES}" "${DEVICES_TYPE}"
  popd > /dev/null
}
[ -f paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl ] || build_paddle
run