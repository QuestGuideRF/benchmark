FILE=$1
echo "Примечание. Доступные модели: Edge2shoes, sat2map, фасады_label2photo и day2night."
echo "Указан [$FILE]"
mkdir -p ./checkpoints/${FILE}_pretrained
MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.pth
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/models-pytorch/$FILE.pth
wget -N $URL -O $MODEL_FILE