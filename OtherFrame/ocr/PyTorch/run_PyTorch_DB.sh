#!/usr/bin/env bash
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.0.2-gpu-cuda10.1-cudnn7"
docker pull ${ImageName}
nvidia-docker stop test_torch_ocr
nvidia-docker rm test_torch_ocr
run_cmd="компакт-диск /рабочая область;
         Баш ПодготовкаEnv.sh;
         cp /workspace/DB_scripts/* /workspace/models/DB/
         компакт-диск /рабочая область/модели/БД/
         pip3.7 install -r require.txt
         bash подготовить_данные.sh
         bash run_benchmark.sh СП
         спать 60
         bash run_benchmark.sh mp
        "
nvidia-docker run --name test_torch_ocr -i \
    --net=host \
    --shm-size=128g \
    -v $PWD:/workspace \
      -v /ssd3:/ssd3 \
    -v /ssd2:/ssd2 \
    -e "ALL_PATH=${all_path}" \
    -v "BENCHMARK_ROOT=/рабочая область" \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${http_proxy}" \
    -e "no_proxy=bcebos.com" \
    ${ImageName} /bin/bash -c "${run_cmd}"
nvidia-docker stop test_torch_ocr
nvidia-docker rm test_torch_ocr