#!/bin/bash

pip install pydub
pip install phonemizer
pip install ffmpeg-normalize
pip install git+https://github.com/wkentaro/gdown.git
git clone -q https://github.com/rmcpantoja/tacotron2.git
cd tacotron2
git clone -q --recursive https://github.com/SortAnon/hifi-gan
pip install git+https://github.com/savoirfairelinux/num2words
git submodule init
git submodule update
pip install matplotlib numpy inflect librosa scipy unidecode pillow tensorboardX
sudo apt-get install pv
sudo apt-get -qq install sox
sudo apt-get install ffmpeg
sudo apt-get install jq
wget https://raw.githubusercontent.com/tonikelope/megadown/master/megadown -O megadown.sh
chmod 755 megadown.sh