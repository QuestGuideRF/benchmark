#!/usr/bin/env bash
ImageName=  ;
docker pull ${ImageName}
run_cmd="Баш ПодготовкаEnv.sh;
        cd /workspace/models/NLP/nlp_modelName/;
        cp /workspace/scripts/NLP/nlp_modelName/preData.sh ./;
        cp /workspace/scripts/NLP/nlp_modelName/run_benchmark.sh ./;
        cp /workspace/scripts/NLP/nlp_modelName/anaанализ_log.py ./;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 32 fp32 500;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh sp 64 fp16 500;
        "
nvidia-docker run --name test_torch -it  \
    --net=host \
    --shm-size=1g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"