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

## Note:
The `--test_only` flag is used to download only the test set videos. It will randomly choose `--num_test` samples from each language. If you want to download the whole dataset, remove the `--test_only` flag. 

Also, the test set would be stored in `os.path.join(args.root, 'test_set')` directory, the other folder structure would be the same as the original dataset.

The official test clips list if provided would be updated later.
