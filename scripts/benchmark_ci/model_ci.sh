#!/бин/баш
cd ${BENCHMARK_ROOT}
if [ -d "logs" ];then rm -rf logs
fi
mkdir logs
cd logs
mkdir static
mkdir dynamic
pip install --upgrade pip
pip install -U opencv-python
pip install tqdm
pip install paddlenlp
export FLAGS_call_stack_level=2
cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
model_list='ResNet50_bs32_dygraph deeplabv3_bs4_fp32 yolov3_bs8 ppyolov2_bs6 ResNet50_pure_fp16_bs64'
source run_models.sh
for model in ${model_list}
do
${model}
done
cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
if [ -f "rerun_model.txt" ];then rm -rf rerun_model.txt
fi
if [ -f "log.txt" ];then rm -rf log.txt
fi
if [ -f "errorcode.txt" ];then rm -rf errorcode.txt
fi
echo success >>log.txt
python analysis.py --log_path=${BENCHMARK_ROOT}/logs/static --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/static --loss_threshold=1 --paddle_dev=False
python analysis.py --log_path=${BENCHMARK_ROOT}/logs/dynamic --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic --loss_threshold=1 --paddle_dev=False
if [ -f "rerun_model.txt" ];then
    echo -e "перезапустите модель в весле, начните разработку!"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.7.0/lib/python3.7/site-packages/paddle/fluid/../libs/
    cd /workspace/Paddle
    pip uninstall -y paddlepaddle_gpu
    pip install build/dev_whl/paddlepaddle_gpu*.whl
    [ $? -ne 0 ] && echo "установить весло не удалось." && exit 1
    mv ${BENCHMARK_ROOT}/logs/static ${BENCHMARK_ROOT}/logs/static_pr
    mv ${BENCHMARK_ROOT}/logs/dynamic ${BENCHMARK_ROOT}/logs/dynamic_pr
    mkdir ${BENCHMARK_ROOT}/logs/static
    mkdir ${BENCHMARK_ROOT}/logs/dynamic
    cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
    model_list=`cat rerun_model.txt`
    echo -e "список повторных моделей: ${model_list}"
    source run_models.sh
    for model in ${model_list}
    do
        ${model}
    done
    cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
    if [ -d "${BENCHMARK_ROOT}/logs/static" ];then
        python analysis_dev.py --log_path=${BENCHMARK_ROOT}/logs/static --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/static
        python analysis.py --log_path=${BENCHMARK_ROOT}/logs/static_pr --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/static --paddle_dev=True
    fi
    if [ -d "${BENCHMARK_ROOT}/logs/dynamic" ];then
        python analysis_dev.py --log_path=${BENCHMARK_ROOT}/logs/dynamic --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic
        python analysis.py --log_path=${BENCHMARK_ROOT}/logs/dynamic_pr --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic --paddle_dev=True
    fi
fi
errorcode='0'
if [ -f "errorcode.txt" ];then
    errorcode=`cat errorcode.txt`
fi
if [[ ${errorcode} != '0' ]];then
    errorcode=`expr $errorcode + 20`
fi
if [[ -z `cat log.txt | grep success` ]];then
    echo -e "Работа по тестированию модели ci не удалась!"
    echo -e "See https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-Model-benchmark-Manual for details."
    echo -e "Или вы можете подать заявку на одобрение QA xiegegege для прохождения этого PR."
    exit ${errorcode}
else
    echo -e "модель эталона ci успех в работе!"
fi