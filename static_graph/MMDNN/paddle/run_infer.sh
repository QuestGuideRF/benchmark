set -e
set -u
set -o pipefail
export CUDA_VISIBLE_DEVICES=1
export FLAGS_fraction_of_gpu_memory_to_use=0.001
in_task_type='predict'
in_conf_file_path='examples/mmdnn-pointwise.json'
python paddle_simnet.py \
		   --task_type $in_task_type \
		   --conf_file_path $in_conf_file_path
endtime=$(date +%Y-%m-%d\ %H:%M:%S)
echo "${endtime}"