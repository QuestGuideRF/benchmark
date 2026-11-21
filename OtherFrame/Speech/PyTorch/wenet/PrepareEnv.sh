test -d venv || virtualenv -p python3.7 venv
source venv/bin/activate
pushd models/wenet
pip install -r requirements.txt
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
popd
cp scripts/executor.py models/wenet/wenet/utils/
VERSION=v4.2.0
BINARY=yq_linux_amd64
wget https://github.com/mikefarah/yq/releases/download/${VERSION}/${BINARY} -O /usr/bin/yq &&\
        chmod +x /usr/bin/yq