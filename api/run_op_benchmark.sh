#!/бин/баш
OP_BENCHMARK_ROOT="$( компакт-диск "$( dirname "${BASH_SOURCE[0]}")" && пароль )"
test_module_name=${1:-"tests"}
gpu_ids=${2:-"0"}
op_type=${3:-"all"}
device_type=${4:-"both"}
precision=${5:-"fp32"}
task=${6:-"both"}
framework=${7:-"both"}
if [ ${test_module_name} != "tests" ] && [ ${test_module_name} != "tests_v2" ]; then
  echo "Установите для параметра test_module_name (${test_module_name}) значение \"tests\", \"tests_v2\"!"
  exit
fi
if [ ${device_type} != "both" ] && [ ${device_type} != "gpu" ] && [ ${device_type} != "cpu" ]; then
  echo "Please set device_type (${device_type}) to \"both\", \"cpu\" or \"gpu\"!"
  exit
fi
if [ ${precision} != "both" ] && [ ${precision} != "fp32" ] && [ ${precision} != "fp16" ]; then
  echo "Please set precision (${precision}) to \"both\", \"fp32\" or \"fp16\"!"
  exit
fi
install_package() {
  local package_name=$1
  local package_version=$2
  echo "-- Модуль сравнительного анализа: ${test_module_name}"
  python -c "import ${package_name}" >/dev/null 2>&1
  import_status=$?
  if [ ${import_status} -eq 0 ]; then
    installed_version=`python -c "import ${package_name}; print(${package_name}.__version__)"`
    if [[ "${installed_version}" > "${package_version}" ]]; then
      echo "-- ${package_name} ${installed_version} (более новая версия, чем ${package_version}) уже установлен."
    elif [ "${installed_version}" == "${package_version}" ]; then
      echo "-- ${package_name} ${package_version} уже установлен."
    else
      if [ "${package_version}" != "" ]; then
        echo "-- Обновление ${package_name}: ${installed_version} -> ${package_version}"
        pip install -U ${package_name}==${package_version}
      else
        pip install -U ${package_name}
      fi
    fi
  else
    echo "-- Установите ${package_name} ${package_version}"
    pip install ${package_name}==${package_version}
  fi
}
run_op_benchmark() {
  local testing_mode=$1
  if [ "${op_type}" = "all" ]; then
    OUTPUT_ROOT=${OP_BENCHMARK_ROOT}/logs
    op_type="None"
  else
    OUTPUT_ROOT=${OP_BENCHMARK_ROOT}/logs/${op_type}
  fi
  if [ ! -d ${OUTPUT_ROOT} ]; then
    mkdir -p ${OUTPUT_ROOT}
  fi
  timestamp=`date '+%Y%m%d-%H%M%S'`
  if [ ${test_module_name} == "tests" ]; then
    subdir=${test_module_name}_${testing_mode}
  else
    subdir=${test_module_name}
  fi
  output_dir=${OUTPUT_ROOT}/${subdir}/${timestamp}
  if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
  fi
  echo "-- выходной_каталог: ${output_dir}"
  config_dir=${OP_BENCHMARK_ROOT}/tests_v2/configs
  echo "--каталог_конфигурации: ${config_dir}"
  tests_dir=${OP_BENCHMARK_ROOT}/${test_module_name}
  echo "--каталог_тестов: ${tests_dir}"
  log_path=${OUTPUT_ROOT}/log_${subdir}_${timestamp}.txt
  bash ${OP_BENCHMARK_ROOT}/deploy/main_control.sh ${tests_dir} ${config_dir} ${output_dir} ${gpu_ids} ${device_type} ${task} "none" ${framework} ${testing_mode} ${op_type} ${precision} > ${log_path} 2>&1 &
}
main() {
  if [ ${test_module_name} == "tests" ]; then
    testing_mode="dynamic"
    install_package "torch" "1.12.0"
    install_package "torchvision"
  else
    testing_mode="static"
    install_package "tensorflow" "2.3.1"
  fi
  run_op_benchmark ${testing_mode}
}
main