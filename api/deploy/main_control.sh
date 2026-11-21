#! /бин/баш
function print_usage() {
    echo "Usage:"
    echo "    bash ${0} test_dir json_config_dir выходной_dir gpu_id процессора|gpu|обе скорости|точность|оба op_list_file framework точность_режима тестирования"
    echo ""
    echo "Arguments:"
    echo "   1. test_dir - каталог тестового кейса"
    echo "   2. json_config_dir — каталог json конфигов"
    echo "   3. output_dir — выходной каталог"
    echo "   4. gpu_id (необязательно) — идентификатор графического процессора. Можно указать только один графический процессор."
    echo "   5. устройство (опционально) — процессор, графический процессор, оба"
    echo "   6. задание (по желанию) – скорость, точность и то, и другое."
    echo "   7. op_list_file (необязательно) — путь, по которому указан список оп для тестирования."
    echo "   8. Фреймворк (необязательно) — весло, тензорный поток, pytorch, оба"
    echo "   9.test_mode (необязательно) — режим тестирования весла, динамический(по умолчанию)|статический."
    echo "  10. имя_операции (необязательно) — указанное имя операции или строка списка. такие как conv2d | conv1d, conv2d, conv3d."
    echo "  11. точность (необязательно) — точность проверки, fp32|fp16|оба"
}
function print_arguments() {
    echo "Arguments:"
    echo "  $*"
    echo ""
    echo "test_dir: ${TEST_DIR}"
    echo "json_config_dir: ${JSON_CONFIG_DIR}"
    echo "выходной_каталог: ${OUTPUT_DIR}"
    echo "gpu_ids: ${GPU_IDS}"
    echo "набор_устройств: ${DEVICE_SET[@]}"
    echo "набор_задач: ${TASK_SET[@]}"
    echo "op_list_file: ${OP_LIST_FILE}"
    echo "структура: ${FRAMEWORK_SET[@]}"
    echo "режим тестирования: ${TESTING_MODE}"
    echo "имя_оператора: ${OP_NAME}"
    echo "точность: ${PRECISION_SET}"
    echo ""
}
declare -A DEVICE_TASK_PID_MAP
declare -A TASK_PID_RUN_START_MAP
declare -A TASK_PID_INFO_MAP
declare -A TASK_PID_DETAIL_CONTENT_MAP
declare -A TASK_PID_DETAIL_KEY_MAP
declare -A DETAIL_KEY_TASK_PIDS_MAP
declare -A DETAIL_KEY_TOTAL_TASK_NUM_MAP
declare -A DETAIL_KEY_FINISHED_TASK_NUM_MAP
declare -A DETAIL_KEY_CONTENT_MAP
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
OP_BENCHMARK_ROOT="$( компакт-диск "$( dirname "${BASH_SOURCE[0]}")/../" && пароль )"
DEPLOY_DIR="${OP_BENCHMARK_ROOT}/развернуть"
export PYTHONPATH=${OP_BENCHMARK_ROOT}:${PYTHONPATH}
if [ $
    print_usage
    exit
fi
TEST_DIR=${1}
TEST_MODULE_NAME=${TEST_DIR
JSON_CONFIG_DIR=${2}
OUTPUT_DIR=${3}
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi
GPU_IDS=${4:-"0"}
GPU_IDS_ARRAY=(${GPU_IDS//,/ })
NUM_GPU_DEVICES=${
for i in ${GPU_IDS_ARRAY[*]}; do
    DEVICE_TASK_PID_MAP[$i]=0
done
if [ ${NUM_GPU_DEVICES} -le 0 ]; then
    echo "Устройства графического процессора (ids=${GPU_IDS}) не должны быть пустыми."
    exit
fi
DEVICE_SET=("gpu" "cpu")
if [ $
    if [[ ${5} == "cpu" || ${5} == "gpu" ]]; then
        DEVICE_SET=(${5})
    fi
fi
TASK_SET=("speed" "accuracy")
if [ $
    if [[ ${6} == "speed" || ${6} == "accuracy" ]]; then
        TASK_SET=(${6})
    fi
fi
if [ $
    OP_LIST_FILE=${7}
    if [ ! -f "${OP_LIST_FILE}" ]; then
        echo "Указанный OP_LIST_FILE (${OP_LIST_FILE}) не существует в файловой системе!"
        unset OP_LIST_FILE
    fi
fi
if [ "${OP_LIST_FILE}" == "" ]; then
    OP_NAME="None"
    if [ $
        OP_NAME=${10}
    fi
    OP_LIST_FILE=${OUTPUT_DIR}/api_info.txt
    python ${DEPLOY_DIR}/collect_api_info.py \
        --test_module_name ${TEST_MODULE_NAME} \
        --specified_op_list ${OP_NAME} \
        --info_file ${OP_LIST_FILE}
    return_status=$?
    if [ ${return_status} -ne 0 ] || [ ! -f "${OP_LIST_FILE}" ]; then
        OP_LIST_FILE=${DEPLOY_DIR}/api_info.txt
    fi
fi
TESTING_MODE="dynamic"
FRAMEWORK_SET=("paddle" "pytorch")
if [ $
    if [ $
        TESTING_MODE=${9}
    elif [ ${TEST_MODULE_NAME} == "tests_v2" ]; then
        TESTING_MODE="static"
    fi
    if [ ${TESTING_MODE} == "static" ]; then
        if [[ ${8} == "paddle"  || ${8} == "tensorflow" ]]; then
            FRAMEWORK_SET=(${8})
        elif [ ${8} == "both" ]; then
            FRAMEWORK_SET=("paddle" "tensorflow")
        else
            echo "В режиме статического тестирования можно тестировать только весло или тензорный поток."
        fi
    elif [ ${TESTING_MODE} == "dynamic" ]; then
        if [[ ${8} == "paddle"  || ${8} == "pytorch" ]]; then
            FRAMEWORK_SET=(${8})
        elif [ ${8} == "both" ]; then
            FRAMEWORK_SET=("paddle" "pytorch")
        else
            echo "В режиме динамического тестирования можно тестировать только весло или pytorch."
        fi
    fi
fi
PRECISION_SET=("fp32")
if [ $
    if [[ ${11} == "fp32" || ${11} == "fp16" ]]; then
        PRECISION_SET=(${11})
    elif [[ ${11} == "both" ]]; then
        PRECISION_SET=("fp32" "fp16")
    fi
fi
function print_detail_status() {
    local config_id=$1
    local case_id=$2
    local device=$3
    local backward=$4
    local logfile=$5
    local gpu_id=$6
    local runtime=$7
    local return_status=$8
    if [ ${backward} = "False" ]; then
        local backward_shorten="F"
    else
        local backward_shorten="T"
    fi
    if  [ ${return_status} -eq 0 ]; then
        local run_status="SUCCESS"
    elif [ ${runtime} -ge 600000 ]; then
        local run_status="**ТАЙМ-АУТ**"
    else
        local run_status="**НЕУСПЕШНЫЙ**"
    fi
    local print_str="устройство=${device}, назад=${backward_shorten}, ${logfile}, время=${runtime} мс"
    local print_str_length=${
    local timestamp=`date +"%Y-%m-%d %T"`
    if [ ${print_str_length} -lt 80 ]; then
        printf "  [%s][%d-%d][%s] %-80s ...... %s\n" "${gpu_id}" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    elif [ ${print_str_length} -lt 100 ]; then
        printf "  [%s][%d-%d][%s] %-100s ...... %s\n" "${gpu_id}" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    else
        printf "  [%s][%d-%d][%s] %-120s ...... %s\n" "${gpu_id}" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    fi
    [ "$BENCHMARK_PRINT_FAIL_LOG" == "1" ] && cat ${logfile}
}
function print_finished_task_detail() {
    local finished_task_pid=$1
    local finished_run_end=$2
    local detail_key=${TASK_PID_DETAIL_KEY_MAP[$finished_task_pid]}
    [ -z "${TASK_PID_INFO_MAP[$finished_task_pid]}" ] && return 0
    finished_run_start=TASK_PID_RUN_START_MAP[$finished_task_pid]
    runtime=$((finished_run_end-finished_run_start))
    runtime=`expr $runtime / 1000000`
    wait $finished_task_pid
    return_status=$?
    if [ ${return_status} -eq 0 ]; then
        num_success_cases=$[$num_success_cases+1]
    else
        num_failed_cases=$[$num_failed_cases+1]
    fi
    if [ ${device} == "gpu" ]; then
        gpu_runtime=`expr $gpu_runtime + $runtime`
    else
        cpu_runtime=`expr $cpu_runtime + $runtime`
    fi
    TASK_PID_DETAIL_CONTENT_MAP[$finished_task_pid]=$(print_detail_status ${TASK_PID_INFO_MAP[$finished_task_pid]} ${runtime} ${return_status})
    DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]=$[${DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]}+1]
    if [ ${DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]} -eq ${DETAIL_KEY_TOTAL_TASK_NUM_MAP[$detail_key]} ]
    then
        detail_content="${DETAIL_KEY_CONTENT_MAP[$detail_key]}"
        for task_pid in ${DETAIL_KEY_TASK_PIDS_MAP[$detail_key]}
        do
            detail_content="${detail_content}\n${TASK_PID_DETAIL_CONTENT_MAP[$task_pid]}"
            unset TASK_PID_DETAIL_CONTENT_MAP[$task_pid]
        done
        echo -e "${detail_content}\n"
        unset DETAIL_KEY_CONTENT_MAP[$detail_key]
        unset DETAIL_KEY_TASK_PIDS_MAP[$detail_key]
        unset DETAIL_KEY_TOTAL_TASK_NUM_MAP[$detail_key]
        unset DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]
    fi
    unset TASK_PID_INFO_MAP[$finished_task_pid]
    unset TASK_PID_RUN_START_MAP[$finished_task_pid]
    unset TASK_PID_DETAIL_KEY_MAP[$finished_task_pid]
}
function execute_one_case() {
    local config_id=$1
    local line=$2
    local json_file_path=$3
    local i=$4
    local api_name=$(echo $line | cut -d ',' -f1)
    local name=$(echo $line | cut -d ',' -f2)
    local has_backward=$(echo $line | cut -d ',' -f4)
    if [ ${has_backward} = False ]; then
        direction_set=("forward")
    else
        direction_set=("forward" "backward")
    fi
    local case_id=0
    local detail_key="${config_id}-${i}"
    local task_pids=""
    DETAIL_KEY_TOTAL_TASK_NUM_MAP[$detail_key]=-1
    DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]=0
    DETAIL_KEY_CONTENT_MAP[$detail_key]="[${config_id}]: api_name=${api_name}, name=${name}, json_file=${json_file_path}, num_configs=${cases_num}, json_id=${i}"
    for device in ${DEVICE_SET[@]}; do
        if [ ${device} = "gpu" ]; then
            local use_gpu="True"
            local repeat=1000
        else
            local use_gpu="False"
            local repeat=100
        fi
        for task in "${TASK_SET[@]}"; do
            for framework in "${FRAMEWORK_SET[@]}"; do
                [ "${task}" == "accuracy" -a "${framework}" == "tensorflow" ] && continue
                [ "${task}" == "accuracy" -a "${framework}" == "pytorch" ] && continue
                for direction in "${direction_set[@]}";
                do
                    if [ ${direction} = "forward" ]; then
                        local backward="False"
                    else
                        local backward="True"
                    fi
                    for precision in "${PRECISION_SET[@]}";
                    do
                        [ "${device}" == "cpu" -a "${precision}" == "fp16" ] && continue
                        case_id=$[$case_id+1]
                        if [ "${TEST_MODULE_NAME}" = "tests" ]; then
                            test_script="${TEST_DIR}/test_main.py --имя файла ${name}"
                        else
                            test_script="${TEST_DIR}/${name}.py"
                        fi
                        run_cmd="python -m common.launch ${test_script} \
                              --api_name ${api_name} \
                              --task ${task} \
                              --framework ${framework} \
                              --testing_mode ${TESTING_MODE} \
                              --json_file ${json_file_path} \
                              --config_id $i \
                              --backward ${backward} \
                              --use_gpu ${use_gpu} \
                              --repeat $repeat \
                              --allow_adaptive_repeat Истина"
                        if [[ "${precision}" == "fp16" ]]; then
                            run_cmd="${run_cmd} --convert_to_fp16 Истина"
                            suffix="_fp16"
                        else
                            suffix=""
                        fi
                        while true
                        do
                            for device_id in ${!DEVICE_TASK_PID_MAP[*]}
                            do
                                task_pid=${DEVICE_TASK_PID_MAP[$device_id]}
                                if [ $task_pid -eq 0 -o -z "$(ps -opid | grep -w $task_pid)" ]
                                then
                                    gpu_id=$device_id
                                    finished_task_pid=$task_pid
                                    break 2
                                fi
                            done
                            sleep 1s
                        done
                        run_start=`date +%s%N`
                        if [ "${OUTPUT_DIR}" != "" ]; then
                            logfile=${OUTPUT_DIR}/${api_name}"_"${i}"-"${framework}"_"${device}"_"${task}"_"${direction}""${suffix}".txt"
                            if [ ${device} = "gpu" ]; then
                                CUDA_VISIBLE_DEVICES="${gpu_id}" timeout 600s ${run_cmd} > $logfile 2>&1 &
                            else
                                CUDA_VISIBLE_DEVICES="" taskset -c ${gpu_id} timeout 600s ${run_cmd} > $logfile 2>&1 &
                            fi
                            task_pid=$!
                        else
                            logfile=""
                            if [ ${device} = "gpu" ]; then
                                CUDA_VISIBLE_DEVICES="${gpu_id}" ${run_cmd} &
                            else
                                CUDA_VISIBLE_DEVICES="" taskset -c ${gpu_id} ${run_cmd} &
                            fi
                            task_pid=$!
                        fi
                        task_pids="${task_pids} ${task_pid}"
                        TASK_PID_RUN_START_MAP[$task_pid]=$run_start
                        DEVICE_TASK_PID_MAP[$gpu_id]=$task_pid
                        TASK_PID_INFO_MAP[$task_pid]="${config_id} ${case_id} ${device} ${backward} ${logfile} ${gpu_id}"
                        TASK_PID_DETAIL_KEY_MAP[$task_pid]=$detail_key
                        [ $finished_task_pid -ne 0 ] && print_finished_task_detail $finished_task_pid $run_start
                    done
                 done
            done
        done
    done
    DETAIL_KEY_TOTAL_TASK_NUM_MAP[$detail_key]=$case_id
    DETAIL_KEY_TASK_PIDS_MAP[$detail_key]="$task_pids"
}
function run_all_cases() {
    local op_info_str=`cat ${OP_LIST_FILE}`
    local op_info_array=(${op_info_str/\\n/ })
    local num_ops=${
    [ -z "$WORKER_NODE_TOTAL" ] && WORKER_NODE_TOTAL=1
    [ -z "$WORKER_NODE_INDEX" ] && WORKER_NODE_INDEX=0
    local num_ops_each_gpu=$((num_ops+WORKER_NODE_TOTAL-1))
    local num_ops_each_gpu=$((num_ops_each_gpu/WORKER_NODE_TOTAL))
    local config_index_begin=$((WORKER_NODE_INDEX*num_ops_each_gpu))
    local config_index_end=$((config_index_begin+num_ops_each_gpu))
    if [ ${config_index_end} -gt ${num_ops} ]; then
        config_index_end=${num_ops}
    fi
    local config_id=0
    echo "config_index_begin: ${config_index_begin}; config_index_end: ${config_index_end};"
    local line_id=${config_index_begin}
    while [ ${line_id} -lt ${config_index_end} ]; do
        local line=${op_info_array[line_id]}
        local case_name=$(echo $line | cut -d ',' -f1)
        local json_file=$(echo $line | cut -d ',' -f3)
        if [ "$json_file" != "None" ]; then
            local json_file_path=${JSON_CONFIG_DIR}/${json_file}
            local cases_num=$(grep '"оп"' ${json_file_path} | wc -l)
        else
            local cases_num=1
            local json_file_path=None
        fi
        if [ -n "$(echo ${case_name} | grep ':')" ]; then
            local case_name_id=${case_name
        else
            local case_name_id=""
        fi
        for((i=0;i<cases_num;i++)); do
            [ -n "${case_name_id}" -a "${case_name_id}" != "${i}" ] && continue
            [ -n "${case_name_id}" ] && line=${line//:${case_name_id}/}
            config_id=$[$config_id+1]
            execute_one_case ${config_id} ${line} ${json_file_path} ${i} ${gpu_id}
        done
        line_id=$((line_id+1))
    done
    while [ ${
    do
        for device_id in ${!DEVICE_TASK_PID_MAP[*]}
        do
            task_pid=${DEVICE_TASK_PID_MAP[$device_id]}
            if [ $task_pid -eq 0 ]
            then
                unset DEVICE_TASK_PID_MAP[$device_id]
            elif [ -z "$(ps -opid | grep -w $task_pid)" ]
            then
                print_finished_task_detail $task_pid $(date +%s%N)
                unset DEVICE_TASK_PID_MAP[$device_id]
            fi
        done
        sleep 1s
    done
}
print_arguments $*
cpu_runtime=0
gpu_runtime=0
num_success_cases=0
num_failed_cases=0
run_all_cases
echo "===================================================================="
echo "Summary:"
echo "  ${num_success_cases} успешно; ${num_failed_cases} не удалось"
echo "  Время работы графического процессора: ${gpu_runtime} мс; Время работы ЦП: ${cpu_runtime} мс"
echo "===================================================================="