if [[ -t 1 ]]
then
    YELLOW="$( эхо -е "\e[33m" )"
    GREEN="$( эхо -е "\e[32m" )"
    RED="$( эхо -е "\e[31m" )"
    NORMAL="$( эхо -е "\e[0m" )"
fi
function _yellow(){ echo "$ ЖЕЛТЫЙ""$@""$НОРМАЛЬНЫЙ";}
function _green(){ echo "$ЗЕЛЕНЫЙ""$@""$НОРМАЛЬНЫЙ";}
function _red(){ echo "$RED""$@""$НОРМАЛЬНЫЙ";}
function _message(){ echo "$@" >&2;}
function _warn(){ echo $(_yellow '==> ПРЕДУПРЕЖДАЕМ:') "$@" >&2;}
function _info(){ echo $(_green '==> ') "$@" >&2;}
function _error(){ echo $(_red '==> ОШИБКА:') "$@" >&2;}
function _fatal(){ echo $(_red '==> ОШИБКА:') "$@" >&2; exit 1;}
function _print_usage(){
    _message "использование: $0 [опции]"
    _message "    -h необязательно Распечатать это справочное сообщение"
    _message "    -b каталог сравнительного_рабочего_пути"
    _message "    -d тип_устройства p40 | v100 | Процессор"
    _message "    -c cuda_версия 9.0|10.0"
    _message "    -n cudnn_версия 7"
    _message "    -разработка image_branch | 1,6 | пиар"
    _message "    -t тип_работы эталонный_далий | тест моделей | pr_test"
    _message "    -p all_path содержит каталог изображений, например /ssd1/ljh."
    _message "    -s тип реализации модели статический | динамичный"
    _message "    -e адрес электронной почты для тестового оповещения"
}
function _init_parameters(){
    benchmark_work_path=$(pwd)
    image_branch='develop'
    device_type=v100
    cuda_version=10.1
    cudnn_version=7
    job_type=12
    all_path=/ssd1/ljh
    implement_type=staticgraph
}
_init_parameters
while [[ $
    case "$1" in
    -h|--help) _print_usage; exit 0 ;;
    -b|--benchmark_work_path)
        [[ "X$2" == "X" ]] && _fatal "вариант $1 нуждается в аргументе!"
        benchmark_work_path=$2
        shift; shift
        ;;
    -d|--device_type)
        [[ "X$2" == "X" ]] && _fatal "вариант $1 нуждается в аргументе!"
        device_type=$2
        shift; shift
        ;;
    -c|--cuda_version)
        [[ "X$2" == "X" ]] && _fatal "вариант $1 нуждается в аргументе!"
        cuda_version=$2
        shift; shift
        ;;
    -n|--cudnn_version)
        [[ "X$2" == "X" ]] && _fatal "вариант $1 нуждается в аргументе!"
        cudnn_version=$2
        shift; shift
        ;;
    -a|--image_branch)
        [[ "X$2" == "X" ]] && _fatal "вариант $1 нуждается в аргументе!"
        image_branch=$2
        shift; shift
        ;;
    -t|--job_type)
        [[ "X$2" == "X" ]] && _fatal "вариант $1 нуждается в аргументе!"
        job_type=$2
        shift; shift
        ;;
    -p|--all_path)
        [[ "X$2" == "X" ]] && _fatal "вариант $1 нуждается в аргументе!"
        all_path=$2
        shift; shift
        ;;
    -s|--implement_type)
        [[ "X$2" == "X" ]] && _fatal "вариант $1 нуждается в аргументе!"
        implement_type=$2
        shift; shift
        ;;
    -e|--email_address)
        [[ "X$2" == "X" ]] && _fatal "вариант $1 нуждается в аргументе!"
        email_address=$2
        shift; shift
        ;;
    *)
        _fatal "Нераспознанный вариант $1"
        ;;
   esac
done
function construnct_version(){
    cd ${benchmark_work_path}/Paddle || exit
    image_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "image_commit_id: "${image_commit_id}
    version=`date -d @$(git log -1 --pretty=format:%ct) "+%Y.%m%d.%H%M%S"`
    image_branch=$(echo ${image_branch} | rev | cut -d'/' -f 1 | rev)
    if [[ ${device_type} == 'cpu' || ${device_type} == "CPU" ]]; then
        PADDLE_VERSION=${version}".${image_branch//-/_}.gcc82"
        IMAGE_NAME=paddlepaddle-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
        with_gpu="OFF"
        cuda_version="10.0"
        cudnn_version=7
    else
        PADDLE_VERSION=${version}'.gcc82.post'$(echo ${cuda_version}|cut -d "." -f1)${cudnn_version}".${image_branch//-/_}"
        IMAGE_NAME=paddlepaddle_gpu-0.0.0.${PADDLE_VERSION}-cp37-cp37m-linux_x86_64.whl
        with_gpu='ON'
    fi
    PADDLE_DEV_NAME=paddlepaddle/paddle_manylinux_devel:cuda${cuda_version}_cudnn${cudnn_version}
    echo "IMAGE_NAME: "${IMAGE_NAME}
}
function build_paddle(){
    construnct_version
    if [[ -e ${all_path}/images/${IMAGE_NAME} ]]
    then
        echo "изображение создано, приступаем к запуску моделей"
        return
    else
        echo "изображение не найдено, начните сборку"
    fi
    PADDLE_ROOT=${PWD}
    docker run -i --rm -v ${PADDLE_ROOT}:/paddle \
      -w /paddle \
      --net=host \
      -e "CMAKE_BUILD_TYPE=Выпуск" \
      -e "PYTHON_ABI=cp37-cp37m" \
      -e "PADDLE_VERSION=0.0.0.${PADDLE_VERSION}" \
      -e "С_AVX=ВКЛ." \
      -e "With_GPU=${with_gpu}" \
      -e "С_TEST=ВЫКЛ." \
      -e "RUN_TEST=ВЫКЛ." \
      -e "С_PYTHON=ВКЛ." \
      -e "With_STYLE_CHECK=ВЫКЛ." \
      -e "С_TESTING=ВЫКЛ." \
      -e "CMAKE_EXPORT_COMPILE_COMMANDS=ВКЛ." \
      -e "С_MKL=ВКЛ." \
      -e "BUILD_TYPE=Выпуск" \
      -e "With_DISTRIBUTE=ВЫКЛ." \
      -e "CMAKE_VERBOSE_MAKEFILE=ВЫКЛ." \
      -e "http_proxy=${HTTP_PROXY}" \
      -e "https_proxy=${HTTP_PROXY}" \
      ${PADDLE_DEV_NAME} \
       /bin/bash -c "весло/скрипты/paddle_build.sh построить $(nproc)"
     build_name=${IMAGE_NAME}
    if [[ -d ${all_path}/images ]]; then
        echo "каталог изображений уже существует"
    else
        mkdir -p ${all_path}/images
    fi
    if [[ -d ${all_path}/logs ]]; then
        echo "каталог журналов уже существует"
    else
        mkdir -p ${all_path}/logs
    fi
    mkdir ./output
    build_link="${CE_SERVER}/viewLog.html?buildId=${BUILD_ID}&buildTypeId=${BUILD_TYPE_ID}&tab=buildLog"
    echo "Ссылка на журнал сборки: ${build_link}"
    if [[ -s ./build/python/dist/${build_name} ]]
    then
        cp ./build/python/dist/${build_name} ${all_path}/images/${IMAGE_NAME}
        echo "построй весло, начни бежать!"
    else
        echo "построить весло не удалось, выйдите!"
        sendmail -t ${email_address} <<EOF
From:paddle_benchmark@test.com
SUBJECT:benchmark运行结果报警, 请检查
Content-type: text/plain
PADDLE BUILD FAILED!!
详情请点击: ${build_link}
EOF
        exit 1
    fi
}
function run(){
    construnct_version
    if [[ -s ${all_path}/images/${IMAGE_NAME} ]]; then echo "изображение найдено"; else exit 1; fi
    logs_dir=${all_path}/logs/logs_op_benchmark/${PADDLE_VERSION}
    mkdir -p ${logs_dir}
    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--устройство {}:{}')
    RUN_IMAGE_NAME=paddlepaddle/paddle:latest-dev-cuda${cuda_version}-cudnn${cudnn_version}-gcc82
    run_cmd="rm -rf /usr/local/python2.7.15/bin/python;
             rm -rf /usr/local/python2.7.15/bin/pip;
             ln -s /usr/local/bin/python3.7 /usr/local/python2.7.15/bin/python;
             ln -s /usr/local/bin/pip3.7 /usr/local/python2.7.15/bin/pip;
             python -m pip install --upgrade pip;
             pip install nvidia-ml-py;
             pip установить psutil;
             pip install tensorflow==2.3;
             pip uninstall Paddlepaddle -y;
             pip uninstall Paddlepaddle-GPU -Y;
             pip install ${all_path}/images/${IMAGE_NAME};
             cd ${benchmark_work_path}/baidu/paddle/benchmark/api;
             bash Deploy/main_control.sh тестирует тесты/конфигурации ${logs_dir} ${CUDA_VISIBLE_DEVICES} gpu;
             bash Deploy/main_control.sh тестирует тесты/конфигурации ${logs_dir} ${CPU_VISIBLE_DEVICES} процессора;
             отключить http_proxy https_proxy;
             python Deploy/post_log.py --server_path ${LOG_SERVER} --file_path ${logs_dir}
             ln -s ${all_path}/env/bin/python /usr/local/bin/mypython;
             экспортировать LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${all_path}/env/lib/;
             mypython Deploy/summary.py ${logs_dir};
            "
    nvidia-docker run ${CUDA_SO} ${DEVICES}  -i --rm \
            -v /home/work:/home/work \
            -v ${all_path}:${all_path} \
            -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
            -v /usr/bin/monquery:/usr/bin/monquery \
            -e "BENCHMARK_WEBSITE1=${BENCHMARK_WEBSITE1}" \
            -e "BENCHMARK_WEBSITE2=${BENCHMARK_WEBSITE2}" \
            -e "http_proxy=${HTTP_PROXY}" \
            -e "https_proxy=${HTTP_PROXY}" \
            -e "PADDLE_VERSION=${PADDLE_VERSION}" \
            -e "RUN_IMAGE_NAME=${RUN_IMAGE_NAME}" \
            -e "PADDLE_COMMIT_ID=${image_commit_id}" \
            -e "START_TIME=$(дата "+%Y%m%d")" \
            -e "BENCHMARK_TYPE=${BENCHMARK_TYPE}" \
            -e "BENCHMARK_GRAPH=${BENCHMARK_GRAPH}" \
            --net=host \
            --privileged \
            --shm-size=32G \
            ${RUN_IMAGE_NAME} \
            /bin/bash -c "${run_cmd}"
}
function send_email(){
    if [[ -e ${logs_dir}/mail.html ]]; then
        cat ${logs_dir}/mail.html |sendmail -t ${email_address}
    fi
}
build_paddle
run
send_email