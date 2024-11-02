import argparse
import json
import os
import cv2
import subprocess
from multiprocessing.pool import ThreadPool
from yt_dlp import YoutubeDL

ANNOTATION_BASE_URL = 'https://github.com/postech-ami/MultiTalk/raw/refs/heads/main/MultiTalk_dataset/annotations/'


class VidInfo:
    def __init__(self, yt_id, time, bbox, raw_vid_dir, processed_vid_dir):
        self.yt_id = yt_id
        self.processed_vid_dir = processed_vid_dir
        self.start_time = float(time[0])
        self.end_time = float(time[1])
        self.video_out_filename = os.path.join(raw_vid_dir, f"{yt_id}.mp4")
        self.bbox = bbox


def process_ffmpeg(raw_vid_path, save_folder, save_vid_name, bbox):
    """
    raw_vid_path:
    save_folder:
    save_vid_name:
    bbox: format: top, bottom, left, right. the values are normalized to 0~1
    """

    def expand(bbox, ratio):
        top, bottom = max(bbox[0] - ratio, 0), min(bbox[1] + ratio, 1)
        left, right = max(bbox[2] - ratio, 0), min(bbox[3] + ratio, 1)

        return top, bottom, left, right

    def to_square(bbox):
        top, bottom, left, right = bbox
        h = bottom - top
        w = right - left
        c = min(h, w) // 2
        c_h = (top + bottom) / 2
        c_w = (left + right) / 2

        top, bottom = c_h - c, c_h + c
        left, right = c_w - c, c_w + c
        return top, bottom, left, right

    def denorm(bbox, height, width):
        top, bottom, left, right = \
            round(bbox[0] * height), \
                round(bbox[1] * height), \
                round(bbox[2] * width), \
                round(bbox[3] * width)

        return top, bottom, left, right

    out_path = os.path.join(save_folder, save_vid_name)

    # get the width and height of the video
    cap = cv2.VideoCapture(raw_vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top, bottom, left, right = to_square(
        denorm(expand(bbox, 0.02), height, width))

    # crop the video and scale to 512x512
    cmd = (f'ffmpeg -i {raw_vid_path} -vf "crop={width * right - width * left}:{height * bottom - height * top}'
           f':{width * left}:{height * top}, scale=512:512" -y {out_path}')
    subprocess.run(cmd, shell=True, check=True)


def download_and_process(vidinfo):
    yt_base_url = 'https://www.youtube.com/watch?v='
    yt_url = yt_base_url + vidinfo.yt_id

    ydl_opts = {
        'format': 'bestvideo+bestaudio',
        'merge_output_format': 'mp4',
        'outtmpl': vidinfo.video_out_filename,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'postprocessor_args': [
            '-y',  # Overwrite output file without asking
            '-ss', f'{vidinfo.start_time}',  # Start time in seconds
            '-to', f'{vidinfo.end_time}',  # End time in seconds
            '-c:v', 'libx264',  # Re-encode video to H.264 for compatibility
            '-c:a', 'aac',  # Re-encode audio to AAC for compatibility
        ],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_url])
    except:
        return_msg = f'{vidinfo.yt_id}, ERROR (youtube)!'
        return return_msg

    try:
        process_ffmpeg(vidinfo.video_out_filename, vidinfo.processed_vid_dir, f'{vidinfo.yt_id}.mp4', vidinfo.bbox)
    except:
        return_msg = f'{vidinfo.yt_id}, ERROR (ffmpeg)!'
        return return_msg

    return_msg = f'{vidinfo.yt_id}, DONE!'
    return return_msg


def load_data(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)

    for key, val in data_dict.items():
        save_name = key + ".mp4"
        ytb_id = val['youtube_id']
        time = val['duration']['start_sec'], val['duration']['end_sec']

        bbox = [val['bbox']['top'], val['bbox']['bottom'],
                val['bbox']['left'], val['bbox']['right']]
        yield ytb_id, save_name, time, bbox


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='+', type=str, default=['english'], help='Languages to download')
    parser.add_argument('--root', type=str, default='./', help='Root folder')
    args = parser.parse_args()

    for language in args.languages:

        processed_vid_root = os.path.join(args.root, 'multitalk_dataset', language)  # processed video path
        raw_vid_root = os.path.join(args.root, 'raw_video', language)  # downloaded raw video path
        os.makedirs(processed_vid_root, exist_ok=True)
        os.makedirs(raw_vid_root, exist_ok=True)
        os.makedirs('./annotation', exist_ok=True)

        # download the annotation file
        annotation_url = ANNOTATION_BASE_URL + f'{language}.json'
        cmd = f'wget {annotation_url} -P ./annotation'
        subprocess.run(cmd, shell=True, check=True)
        json_path = f'./annotation/{language}.json'

        vidinfos = [VidInfo(ytb_id, time, bbox, raw_vid_root, processed_vid_root)
                    for ytb_id, save_name, time, bbox in load_data(json_path)]

        bad_files = open(f'bad_files_{language}.txt', 'w')
        results = ThreadPool(10).imap_unordered(download_and_process, vidinfos)

        cnt, err_cnt = 0, 0
        for r in results:
            cnt += 1
            print(cnt, '/', len(vidinfos), r)
            if 'ERROR' in r:
                bad_files.write(r + '\n')
                err_cnt += 1

        bad_files.close()
        print("Total Error : ", err_cnt)

        # Optionally delete the raw video files
        # cmd = f'rm -rf {raw_vid_root}'
        # subprocess.run(cmd, shell=True, check=True)
