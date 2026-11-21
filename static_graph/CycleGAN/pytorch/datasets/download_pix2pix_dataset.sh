FILE=$1
if [[ $FILE != "cityscapes" && $FILE != "night2day" && $FILE != "edges2handbags" && $FILE != "edges2shoes" && $FILE != "facades" && $FILE != "maps" ]]; then
  echo "Доступные наборы данных: городские пейзажи, ночь2день, края2сумки, края2обувь, фасады, карты."
  exit 1
fi
echo "Указан [$FILE]"
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE