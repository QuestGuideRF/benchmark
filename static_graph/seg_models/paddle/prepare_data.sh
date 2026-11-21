set -ex
mkdir -p  data/cityscape
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip
mv gtFine data/cityscape
mv leftImg8bit data/cityscape
git clone https://github.com/mcordts/cityscapesScripts.git
cp -r cityscapesScripts/cityscapesscripts data/cityscape
export PYTHONPATH=$PWD/data/cityscape
python data/cityscape/cityscapesscripts/preparation/createTrainIdLabelImgs.py
wget https://paddle-deeplab.bj.bcebos.com/deeplabv3plus_xception65_initialize.tgz
tar -xf deeplabv3plus_xception65_initialize.tgz && rm deeplabv3plus_xception65_initialize.tgz