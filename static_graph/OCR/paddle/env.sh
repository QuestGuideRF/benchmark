#!/бин/баш
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
ht=`lscpu |grep "на ядро"|awk -F':' '{print $2}'|xargs`
if [ $ht -eq 1 ]; then
  if [ -z "$KMP_AFFINITY" ]; then
    export KMP_AFFINITY="степень детализации = мелкая, компактная, 0,0"
  fi
  if [ -z "$OMP_DYNAMIC" ]; then
    export OMP_DYNAMIC="FALSE"
  fi
else
  if [ -z "$KMP_AFFINITY" ]; then
    export KMP_AFFINITY="степень детализации = мелкая, компактная, 1,0"
  fi
fi