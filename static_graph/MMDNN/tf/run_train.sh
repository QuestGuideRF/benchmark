set -e
set -u
set -o pipefail
echo "конвертировать данные поезда"
python ./tools/tf_record_writer.py pointwise ./data/train_pointwise_data ./data/convert_train_point_data 0 32
echo "конвертировать тестовые данные"
python ./tools/tf_record_writer.py pointwise ./data/test_pointwise_data ./data/convert_test_point_data 0 32
echo "конвертировать данные закончить"
in_task_type='train'
in_task_conf='./examples/mmdnn-pointwise.json'
python tf_simnet.py \
		   --task $in_task_type \
		   --task_conf $in_task_conf