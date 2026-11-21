#!/бин/баш
if [ $
  echo "Usage: "
  echo "  Конфигурация задачи sh run.sh"
  echo "Например:"
  echo "  sh run.sh поезд train_gpu_perf_1"
  exit
fi
TASK=$1
CONFIG=$2
if [ ${TASK} == "train" ]; then
  case ${CONFIG} in
    train_gpu_perf_1)
      ;;
    train_gpu_perf_8)
      ;;
    train_gpu_memory_1)
      ;;
    train_gpu_memory_8)
      ;;
    *)
      echo "Когда задача — обучение, конфигурация должна быть train_gpu_perf_1/8 или train_gpu_memory_1/8."
      exit
  esac
elif [ ${TASK} == "infer" ]; then
  case ${CONFIG} in
    infer_gpu_perf)
      ;;
    infer_gpu_memory)
      ;;
    infer_cpu)
      ;;
    *)
      echo "Когда задача является выводом, конфигурация должна быть infer_gpu_perf, infer_gpu_memory или infer_cpu."
      exit
    esac
else
  echo "Задача должна быть либо обучением, либо выводом"
  exit
fi
export OCR_work_root="$( компакт-диск "$( dirname "${BASH_SOURCE[0]}")/.." && пароль )"
export OCR_model="attention"
if [ ${OCR_model} == "crnn_ctc" ]; then
  INIT_MODEL_PACK_NAME=ocr_ctc
  export OCR_init_model=${OCR_work_root}/paddle/${INIT_MODEL_PACK_NAME}/ocr_ctc_params
elif [ ${OCR_model} == "attention" ]; then
  INIT_MODEL_PACK_NAME=ocr_attention
  export OCR_init_model=${OCR_work_root}/paddle/${INIT_MODEL_PACK_NAME}/ocr_attention_params
fi
if [ ! -f ${OCR_init_model} ]; then
  echo ">>> Загрузите предварительно подготовленные параметры для модели ${OCR_model}..."
  wget https://paddle-ocr-models.bj.bcebos.com/${INIT_MODEL_PACK_NAME}.zip -O ${OCR_work_root}/paddle/${INIT_MODEL_PACK_NAME}.zip
  unzip ${INIT_MODEL_PACK_NAME}.zip
fi
source ${OCR_work_root}/download.sh
source configs/${CONFIG}.sh
bash ${TASK}.sh