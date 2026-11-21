#!/usr/bin/env bash
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7"
docker pull ${ImageName}
nvidia-docker stop test_torch_gpt
nvidia-docker rm test_torch_gpt
run_cmd="
        установить -xe;
        bash /workspace/PrepareEnv.sh;
        компакт-диск /workspace/models/NLP/gpt/;
        cp /workspace/scripts/NLP/gpt/preData.sh ./;
        cp /workspace/scripts/NLP/gpt/run_benchmark.sh ./;
        СР/рабочая область/анализ.py ./;
        ш preData.sh;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 200;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp16 200;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 16 fp16 200;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 8 fp32 200;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 8 fp16 200;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 16 fp16 200;
        "
nvidia-docker run --name test_torch_gpt -i  \
    --net=host \
    --shm-size=1g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"
nvidia-docker stop test_torch_gpt
nvidia-docker rm test_torch_gpt