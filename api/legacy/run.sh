#!/бин/баш
[ -z "$(set | grep '^CUDA_VISIBLE_DEVICES=')" ] && export CUDA_VISIBLE_DEVICES="0"
NVCC=`which nvcc`
if [ ${NVCC} != "" ]; then
  NVCC_VERSION=`nvcc --version | tail -n 1 | grep "[0-9][0-9]*\.[0-9]" -o | uniq`
  export LD_LIBRARY_PATH=/usr/local/cuda-${NVCC_VERSION}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
fi
OP_BENCHMARK_ROOT="$( компакт-диск "$( dirname "${BASH_SOURCE[0]}")/../" && пароль )"
export PYTHONPATH=${OP_BENCHMARK_ROOT}:${PYTHONPATH}
name=${1:-"abs"}
config_id=${2:-"0"}
task=${3:-"accuracy"}
framework="paddle"
filename="${OP_BENCHMARK_ROOT}/legacy/configs/${name}.json"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  use_gpu=False
else
  use_gpu=True
fi
run_args="--task ${task} \
          --framework ${framework} \
          --testing_mode статический \
          --json_file ${filename} \
          --config_id ${config_id} \
          --profiler нет \
          --backward Истина \
          --use_gpu ${use_gpu} \
          --повторить 1 \
          --allow_adaptive_repeat Ложь \
          --log_level 0"
if [ $
  api_name=${4}
  run_args="${run_args} \
            --api_name ${api_name}"
fi
python -m common.launch ${OP_BENCHMARK_ROOT}/legacy/${name}.py \
         ${run_args}