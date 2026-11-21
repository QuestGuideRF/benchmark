#!/бин/баш
set -xe
gpu_id="-1"
if [ $
  gpu_id="$1"
fi
num_threads=1
if [ $
  num_threads=$2
fi
if [ ${gpu_id} -eq "-1" ]; then
  USE_GPU=false
  export CUDA_VISIBLE_DEVICES=""
else
  USE_GPU=true
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi
MODEL_DIR=${fp32_model_dir}
DATA_FILE=${dataset_dir}/1.8w.bs1
REPEAT=1
if [ $
  MODEL_DIR=$3
fi
if [ $
  DATA_FILE=$4
fi
profile=false
if [ $
  profile=$5
fi
print_outputs=false
if [ $
  print_outputs=$6
fi
GLOG_logtostderr=1 ./build/inference \
    --model_dir=${MODEL_DIR} \
    --data=${DATA_FILE} \
    --repeat=${REPEAT} \
    --warmup_steps=1 \
    --use_gpu=${USE_GPU} \
    --num_threads=${num_threads} \
    --use_analysis=true \
    --print_outputs=${print_outputs} \
    --profile=${profile}