#!/бин/баш
CUR_DIR=${PWD}
source venv/bin/activate
pushd models/wenet/examples/aishell/s0/
mkdir -p exp/log
. path.sh
bash run.sh --data $PWD --stop_stage 3
model_mode_list=(conformer)
fp_item_list=(fp32)
bs_item_list=(16)
for model_mode in ${model_mode_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_item_list[@]}
            do
            rm exp -rf
            echo "индекс – скорость, 1 ГБ, начало, ${model_name}"
            run_mode=sp
            CUDA_VISIBLE_DEVICES=0 bash ${CUR_DIR}/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
            sleep 60
            rm exp -rf
            echo "индекс — скорость, 8 ГБ, run_mode — multi_process, начало, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ${CUR_DIR}/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
            sleep 60
            done
      done
done
popd
mkdir -p log
bash run_analysis_sp.sh > log/log_sp.out
bash run_analysis_mp.sh > log/log_mp.out