#!/usr/bin/env bash
root_dir=${ROOT_DIR:-"/workspace"}
all_path=${all_path}
log_path_index_dir=${LOG_PATH_INDEX_DIR:-$(pwd)}
train_log_dir=${TRAIN_LOG_DIR:-$(pwd)}
run_plat=${RUN_PLAT:-"local"}
run_cmd="компакт-диск /рабочая область;
         bash test_clas.sh"
ImageName="paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82";
docker pull ${ImageName}
nvidia-docker stop test_pytorch_clas
nvidia-docker rm test_pytorch_clas
nvidia-docker run  -i --rm  \
    --name test_pytorch_clas \
    --net=host \
    --cap-add=ALL \
    --shm-size=64g \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${https_proxy}" \
    -e ROOT_DIR=${root_dir} \
    -e LOG_PATH_INDEX_DIR=${log_path_index_dir} \
    -e TRAIN_LOG_DIR=${train_log_dir} \
    -e RUN_PLAT=${run_plat} \
    -e all_path=${all_path} \
    -v $PWD:/workspace \
    -v /ssd3:/ssd3 \
    -v /ssd2:/ssd2 \
    ${ImageName}  \
    /bin/bash -c "${run_cmd}"
nvidia-docker stop test_pytorch_clas
nvidia-docker rm test_pytorch_clas