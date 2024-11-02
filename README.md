# A Multiprocess Downloader for MultiTalk Dataset
Original dataset: [MultiTalk](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset)

## Instal requirements
Install ffmpeg, for example:
```shell
conda install -c conda-forge ffmpeg
# or
sudo apt-get install ffmpeg
```
```bash
pip install -r requirements.txt
```

## Usage
```bash
python download_and_preprocess_mp.py \
--languages english italian french greek  \
--root /datasets/MultiTalk \
--test_only --num_test 5
```
