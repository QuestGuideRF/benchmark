#!/usr/bin/env bash
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7"
docker pull ${ImageName}
nvidia-docker stop test_torch_xlnet
nvidia-docker rm test_torch_xlnet
run_cmd="bash /workspace/PrepareEnv_xlnet.sh;
        компакт-диск /workspace/models/xlnet/;
        cp /workspace/scripts/xlnet/run_benchmark.sh ./;
        cp /workspace/scripts/xlnet/anaанализ_log.py ./;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 32 fp32 1000 xlnet-base-cased;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 128 fp32 1000 xlnet-base-cased;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 32 fp32 1000 xlnet-base-cased;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 128 fp32 1000 xlnet-base-cased;
        "
nvidia-docker run --name test_torch_xlnet -i  \
    --net=host \
    --shm-size=1g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"
nvidia-docker stop test_torch_xlnet
nvidia-docker rm test_torch_xlnet