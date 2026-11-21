FILE=$1
echo "Примечание: доступны модели apple2orange, Orange2Apple, Summer2Winter_Yosemite, Winter2Summer_Yosemite, Horse2zebra, zebra2horse, monet2photo, style_monet, style_cezanne, style_ukiyoe, style_vangogh, sat2map, Map2sat, Cityscapes_photo2label, Cityscapes_label2photo, Fasadas_photo2label, Fasadas_label2photo, iphone2dslr_flower"
echo "Указан [$FILE]"
mkdir -p ./checkpoints/${FILE}_pretrained
MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.pth
URL=http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/$FILE.pth
wget -N $URL -O $MODEL_FILE