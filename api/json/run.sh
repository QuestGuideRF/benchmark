#!/бин/баш
INPUT_DIR=$1
OUTPUT_DIR=$2
unset GREP_OPTIONS
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi
declare -A SPECIAL_DICTS
SPECIAL_DICTS=( \
    ["accuracy"]="k правильный итог" \
    ["affine_channel"]="act" \
    ["anchor_generator"]="anchor_sizes" \
    ["assign"]="output" \
    ["batch_norm"]="эпсилон is_test действовать в_месте импульс" \
    ["conv2d"]="действие номер_фильтров use_cudnn" \
    ["conv2d_transpose"]="действие номер_фильтров use_cudnn" \
    ["depthwise_conv2d"]="действовать группы num_filters" \
    ["dropout"]="is_test семя dropout_prob" \
    ["embedding"]="дополнение_idx is_sparse" \
    ["fc"]="size" \
    ["fill_constant"]="значение Force_cpu" \
    ["instance_norm"]="epsilon" \
    ["l2_normalize"]="epsilon" \
    ["layer_norm"]="действовать эпсилон" \
    ["leaky_relu"]="alpha" \
    ["matmul"]="alpha" \
    ["pool2d"]="use_cudnn" \
    ["reshape"]="форма на месте" \
    ["roi_align"]="spatial_scale" \
    ["scale"]="закон о смещении масштаба" \
    ["softmax_with_cross_entropy"]="return_softmax ignore_index" \
    ["transpose"]="perm" \
)
declare -A SIMILAR_API
SIMILAR_API=( \
    ["activation"]="abs cos exp пол relu sin sigmoid softsign Square sqrt tanh" \
    ["arg"]="аргументмакс аргументмин" \
    ["compare"]="меньше_чем меньше_равно не_равно больше_чем больше_равно равно" \
    ["elementwise"]="elementwise_add elementwise_div elementwise_max elementwise_min elementwise_mul elementwise_sub elementwise_pow" \
    ["logical"]="логическое_и логическое_или" \
    ["reduce"]="уменьшить_среднее уменьшить_сумму уменьшить_прод" \
)
filenames=$(ls ${INPUT_DIR}/*.json)
for file in ${filenames}
do
    op_name=$(basename ${file} .json)
    skipped=0
    for key in $(echo ${!SIMILAR_API[*]})
    do
        result=$(echo ${SIMILAR_API[${key}]} | grep -w "${op_name}")
        if [[ ${result} != "" ]]; then
            skipped=1
            break
        fi
    done
    if [[ ${skipped} -eq 1 ]]; then
        continue
    fi
    append_params=''
    for key in $(echo ${!SPECIAL_DICTS[*]})
    do
        if [[ ${key} = ${op_name} ]]; then
            append_params=${SPECIAL_DICTS[${key}]}
            break
        fi
    done
    echo "API обработки:" ${op_name}
    python2 select_configs.py --input_json_file ${file} --output_json_file ${OUTPUT_DIR}/${op_name}.json --ignored_params ${append_params}
    echo ""
done
for key in $(echo ${!SIMILAR_API[*]})
do
    echo "API обработки:" ${SIMILAR_API[${key}]}
    python2 select_configs.py --input_json_file ${INPUT_DIR} --output_json_file ${OUTPUT_DIR}/${key}.json --similar_api ${SIMILAR_API[${key}]} --ignored_params "act"
    echo ""
done