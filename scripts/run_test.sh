#!/usr/bin/env bash
set -ex
if [ -z ${BRANCH} ]; then
    BRANCH="master"
fi
BENCHMARK_ROOT="$( компакт-диск "$( dirname "${BASH_SOURCE[0]}")/.." && пароль )"
echo ${BENCHMARK_ROOT}
function prepare_env(){
    easy_install --upgrade pip
    PADDLE_WHL="paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl"
    PADDLE_URL="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/${PADDLE_WHL}"
    wget -q ${PADDLE_URL}
    pip install -U ${PADDLE_WHL}
    pip install tensorflow-gpu==2.0 pre-commit==1.21 pylint==1.9.5 pytest==4.6.9
    python -c "import tensorflow as tf; print(tf.__version__)"
    apt-get update
    apt-get install -y git
}
function fetch_upstream_master_if_not_exist() {
    UPSTREAM_URL='https://github.com/PaddlePaddle/benchmark'
    origin_upstream_url=`git remote -v | awk '{print $1, $2}' | uniq | grep upstream | awk '{print $2}'`
    if [ "$origin_upstream_url" == "" ]; then
        git remote add upstream $UPSTREAM_URL.git
    elif [ "$origin_upstream_url" != "$UPSTREAM_URL" ] \
            && [ "$origin_upstream_url" != "$UPSTREAM_URL.git" ]; then
        git remote remove upstream
        git remote add upstream $UPSTREAM_URL.git
    fi
    if [ ! -e "$PADDLE_ROOT/.git/refs/remotes/upstream/$BRANCH" ]; then
        git fetch upstream
    fi
}
function run_api(){
    fetch_upstream_master_if_not_exist
    HAS_MODIFIED_API_TEST=`git diff --name-status upstream/$BRANCH | awk '1 доллар!="Д" {print $2}' | grep "API/tests_v2.*.py$" || true`
    API_NAMES=(abs)
    if [ "${HAS_MODIFIED_API_TEST}" != "" ] ; then
        for api in ${HAS_MODIFIED_API_TEST[@]}; do
            new_name=`echo $api |awk -F "/" '{print $NF}' |awk -F "." '{print $NR}'`
            if [[ "$new_name" != "main" && "$new_name" != "common_ops" && "$new_name" != "launch" ]]; then
                need_append="yes"
                for name in ${API_NAMES[@]}; do
                    if [ "${name}" == "${new_name}" ]; then
                        need_append="no"
                        break
                    fi
                done
                if [ "$need_append" == "yes" ]; then
                    API_NAMES[${
                fi
            fi
        done
    fi
    fail_name=()
    for name in ${API_NAMES[@]}; do
        bash ${BENCHMARK_ROOT}/api/tests_v2/run.sh $name -1
        if [ $? -ne 0 ]; then
            fail_name[${
        fi
    done
    len=${
    if [ $len -ne 0 ]; then
        echo "Неудачные ТЕСТЫ API: ${fail_name[@]}"
        exit 1
    fi
}
function abort(){
    echo "Ваше изменение не соответствует стилю кода теста." 1>&2
    echo "Пожалуйста, используйте предварительную фиксацию, чтобы проверить, что не так." 1>&2
    exit 1
}
function check_style(){
	trap 'abort' 0
	pre-commit install
	commit_files=on
    	for file_name in `git diff --numstat | awk '{print $NF}'`;do
        	if [ ! pre-commit run --files $file_name ]; then
            		git diff
            		commit_files=off
        	fi
    	done
    	if [ $commit_files == 'off' ];then
        	echo "ошибка формата кода"
        	exit 1
    	fi
    	trap 0
}
function build_and_test(){
    apt_mirror='s#http://archive.ubuntu.com/ubuntu#mirror://mirrors.ubuntu.com/mirrors.txt#g'
    PADDLE_DEV_NAME='hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.0-cudnn7'
    docker pull ${PADDLE_DEV_NAME}
    nvidia-docker run --net=host $SHM -i --rm -v $PWD:/benchmark -w /benchmark \
        -v ${GIT_PATH}:${GIT_PATH} \
        -e "APT_MIRROR=${apt_mirror}" \
        -e "http_proxy=${http_proxy}" \
        -e "https_proxy=${https_proxy}" \
        -e "GITHUB_API_TOKEN=${github_api_token}" \
        -e "ФИЛИАЛ=${branch}" \
        ${PADDLE_DEV_NAME} \
        bash scripts/run_test.sh run_api_test
}
function main(){
    local CMD=$1
    case $CMD in
      run_api_test)
        prepare_env
        check_style
        run_api
        ;;
      build_test)
        build_and_test
        ;;
	*)
        echo "Извините, $CMD не распознан."
        exit 1
        ;;
      esac
      echo "сценарий runtest завершился, как и ожидалось"
}
main $@