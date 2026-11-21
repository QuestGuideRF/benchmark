#!bin/баш
if [ $
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|тестовая скорость|mem /ssd3/benchmark_results/cwh/logs"
    exit
fi
task="$1"
index="$2"
run_log_path=${3:-$(pwd)}
model_name="AttGAN"
device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${
batch_size=32
log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}
train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    train_cmd=" --img_size 128 \
        --shortcut_layers 1 \
        --inject_layers 1 \
        --experiment_name 128_shortcut1_inject1_none \
        --gpu"
    train_cmd="python -u поезд.py "${train_cmd}
    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 900
    kill -9 $train_pid
}
analysis_times(){
    skip_step=$1
    awk 'BEGIN{count=0}/Время:/{
      count_fields=NF;
      step_times[count]=$count_fields;
      count+=1;
    }END{
      print "\n================ Результат теста ================"
      напечатайте "total_step:", подсчитайте
      напечатайте "batch_size:", "'${batch_size}'"
      if(count>1){
        step_latency=0
        step_latency_without_step0_avg=0
        step_latency_without_step0_min=step_times['${skip_step}']
        Step_latency_without_step0_max=step_times['${skip_step}']
        for(i=0;i<count;++i){
          step_latency+=step_times[i];
          if(i>='${skip_step}'){
            step_latency_without_step0_avg+=step_times[i];
            if(step_times[i]<step_latency_without_step0_min){
              step_latency_without_step0_min=step_times[i];
            }
            if(step_times[i]>step_latency_without_step0_max){
              step_latency_without_step0_max=step_times[i];
            }
          }
        }
        step_latency/=count;
        step_latency_without_step0_avg/=(count-'${skip_step}')
        printf("Средняя задержка (исходный результат):\n")
        printf("\tAvg: %.3f s/step\n", step_latency)
        printf("\tFPS: %.3f изображений/с\n", "'${batch_size}'"/step_latency)
        printf("средняя задержка (пропустить '${skip_step}' шаги):\n")
        printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
        printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
        printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
        printf("\tFPS: %.3f изображений/с\n", '${batch_size}'/step_latency_without_step0_avg)
        printf("\n")
      }
    }' ${log_file}
}
echo "Тест для $task"
if [ $index = "mem" ]
then
    export FLAGS_fraction_of_gpu_memory_to_use=0.001
    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    $task
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
else
    job_bt=`date '+%Y%m%d%H%M%S'`
    $task
    job_et=`date '+%Y%m%d%H%M%S'`
    hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
    cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
    gpu_num=$(nvidia-smi -L|wc -l)
    if [ ${task} = "train" ]
    then
      analysis_times 3
    else
      echo "нет тестового cmd"
    fi
fi