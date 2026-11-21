#!bin/баш
set -xe
if [ $
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|вывести скорость|mem|maxbs /ssd1/ljh/logs"
  exit
fi
task="$1"
index="$2"
run_log_path=${3:-$(pwd)}
model_name="BiDAF"
device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${
batch_size=32
log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}
train(){
  echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
  echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
  train_cmd="python run.py --train --algo BIDAF --epochs 10 \
    --train_files ../data/extracted/trainset/zhidao.train.json ../data/extracted/trainset/search.train.json \
    --dev_files ../data/extracted/devset/zhidao.dev.json ../data/extracted/devset/search.dev.json \
    --test_files ../data/extracted/testset/zhidao.test.json ../data/extracted/testset/search.test.json"
  ${train_cmd} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 1800
  kill -9 `ps -ef|grep python |awk '{print $2}'`
}
infer(){
  echo "сделать вывод о графических процессорах ${num_gpu_devices}"
  echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
}
analysis_times(){
  skip_step=$1
  awk 'BEGIN{count=0}/скорость:/{
    count_fields=NF-1
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
      printf("\tAvg: %.3f шагов/с\n", step_latency)
      printf("\tFPS: %.3f примеров/с\n", "'${batch_size}'"*step_latency)
      printf("средняя задержка (пропустить '${skip_step}' шаги):\n")
      printf("\tAvg: %.3f шагов/с\n", step_latency_without_step0_avg)
      printf("\tMin: %.3f шагов/с\n", step_latency_without_step0_min)
      printf("\tMax: %.3f шагов/с\n", step_latency_without_step0_max)
      printf("\tFPS: %.3f примеров/с\n", '${batch_size}'*step_latency_without_step0_avg)
      printf("\n")
    }
  }' ${log_file}
}
echo "Тест для $task"
if [ $index = "mem" ]
then
    if [ `grep -c "sess_config.gpu_options.allow_growth" rc_model.py` -gt 0 ]; then
      echo "Найденный!"
    else
      sed -i '/sess_config = tf.ConfigProto()/a\        sess_config.gpu_options.allow_growth = True' rc_model.py
    fi
    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    $task
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
elif [ ${index} = 'speed' ]
then
    sed -i '/sess_config.gpu_options.allow_growth/d' rc_model.py
    $task
    if [ ${task} = "train" ]
    then
      analysis_times 0
    else
      echo "нет вывода cmd"
    fi
else
  $task
  error_string="Невозможно выделить"
  if [ `grep -c "${error_string}" ${log_parse_file}` -eq 0 ]; then
    echo "maxbs: ${batch_size}"
  else
    echo "ошибка запуска maxbs"
  fi
fi