#!/бин/баш
cmd="python3 $1.py --dataroot ./datasets/horse2zebra --namehorse2zebra_cyclegan --modelcycle_gan"
if [ $1 == "train" ]; then
  cmd=${cmd}' --display_id -1'
fi
echo $
if [ $
  cmd=${cmd}" --gpu_ids $2"
fi
echo $cmd
${cmd}