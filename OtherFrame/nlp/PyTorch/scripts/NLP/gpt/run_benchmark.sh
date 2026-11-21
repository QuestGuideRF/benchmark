#!/usr/bin/env bash
set -xe
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}
    max_iter=${4}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    model_name="nlp_gpt"
    mission_name="语义表示"
    direction_id=1
    ips_unit="tokens/s"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${
    model_name=${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    log_file=${run_log_path}/${model_name}
    index_log_file=${run_log_path}/${model_name}_speed
    CHECKPOINT_PATH=${run_log_path}/results/checkpoints
    DATA_PATH=${run_log_path}/data/my-gpt2_text_document
    TOKEN_FILES=${run_log_path}/token_files
    export PATH=/workspace/run_env:${PATH}
    echo `python3-config --help`
}
function _train(){
    echo "Тренируйтесь на графических процессорах ${num_gpu_devices}"
    echo "текущий CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, patch_size=$batch_size"
    if [ -d $CHECKPOINT_PATH ]; then
        rm -rf $CHECKPOINT_PATH
    fi
    if [ $fp_item = "fp16" ]; then
        use_fp16_cmd="--fp16"
    fi
    train_cmd="\
       --num-layers 12 \
       --hidden-size 768\
       --num-attention-heads 12\
       --micro-batch-size $batch_size \
       --global-batch-size $(($batch_size*$num_gpu_devices)) \
       --seq-длина 1024 \
       --max-position-embeddings 1024 \
       --train-iters $max_iter\
       --lr-decay-iters 320000\
       --save $CHECKPOINT_PATH \
       --загрузить $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $TOKEN_FILES/gpt-en-vocab.json \
       --merge-file $TOKEN_FILES/gpt-en-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style косинус \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-интервал 1 \
       --save-interval 2000 \
       --eval-интервал 500 \
       --eval-iters 10 \
       $use_fp16_cmd
    "
    DISTRIBUTED_ARGS="--nproc_per_node $num_gpu_devices --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6010"
    case ${run_mode} in
    sp) train_cmd=" python -u pretrain_gpt.py "${train_cmd} ;;
    mp)
        train_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py "${train_cmd} ;;
    *) echo "выберите run_mode (sp или mp)"; exit 1;
    esac
    cd Megatron-LM
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, ОШИБКА"
        export job_fail_flag=1
    else
        echo -e "${model_name}, УСПЕХ"
        export job_fail_flag=0
    fi
    cd -
}
function _analysis_log(){
    python analysis.py --filename ${log_file} \
        --keyword "(РС):" \
        --base_batch_size $(($batch_size*1024)) \
        --skip_steps 20 \
        --model_mode -2 \
        --model_name ${model_name}\
        --mission_name ${mission_name}\
        --direction_id ${direction_id} \
        --run_mode ${run_mode} \
        --gpu_num ${num_gpu_devices} \
        --ips_unit ${ips_unit} \
        --time_unit 'ms' > ${index_log_file}
}
_set_params $@
_train
_analysis_log