import os
import json
import cv2
import argparse
import subprocess
import tqdm
from yt_dlp import YoutubeDL

ANNOTATION_BASE_URL = 'https://github.com/postech-ami/MultiTalk/raw/refs/heads/main/MultiTalk_dataset/annotations/'

VALID_LANGUAGES = ['arabic', 'catalan', 'croatian', 'czech', 'dutch', 'english', 'french', 'german', 'greek',
                   'hindi', 'italian', 'japanese', 'mandarin', 'polish', 'portuguese', 'russian',
                   'spanish', 'thai', 'turkish', 'ukrainian']


def download_video(yt_id, raw_vid_dir):
    url = f'https://www.youtube.com/watch?v={yt_id}'
    video_path = os.path.join(raw_vid_dir, f"{yt_id}.mp4")

    # Skip download if video already exists
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        return f"{yt_id} already downloaded."

    ydl_opts = {
        'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a][acodec^=mp4a]',
        'merge_output_format': 'mp4',
        'outtmpl': video_path,
        'retries': 3,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'postprocessor_args': [
            '-c:v', 'copy',  # Copy video stream without re-encoding
            '-c:a', 'copy',  # Copy audio stream without re-encoding
        ],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"{yt_id} downloaded successfully."
    except Exception as e:
        return f"{yt_id}, ERROR: {str(e)}"


def process_ffmpeg(raw_vid_path, save_folder, save_vid_name, bbox, time):
    def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min), int(sec), int(end))

    def expand(bbox, ratio):
        top, bottom = max(bbox[0] - ratio, 0), min(bbox[1] + ratio, 1)
        left, right = max(bbox[2] - ratio, 0), min(bbox[3] + ratio, 1)
        return top, bottom, left, right

    def to_square(bbox):
        top, bottom, left, right = bbox
        h = bottom - top
        w = right - left
        c = min(h, w) / 2
        c_h = (top + bottom) / 2
        c_w = (left + right) / 2
        top, bottom = c_h - c, c_h + c
        left, right = c_w - c, c_w + c
        return top, bottom, left, right

    def denorm(bbox, height, width):
        top, bottom, left, right = (
            round(bbox[0] * height),
            round(bbox[1] * height),
            round(bbox[2] * width),
            round(bbox[3] * width)
        )
        return top, bottom, left, right

    out_path = os.path.join(save_folder, save_vid_name)
    cap = cv2.VideoCapture(raw_vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    top, bottom, left, right = to_square(denorm(expand(bbox, 0.02), height, width))
    start_sec, end_sec = time

    cmd = f"ffmpeg -i {raw_vid_path} -r 25 -vf crop=w={right - left}:h={bottom - top}:x={left}:y={top},scale=512:512 -ss {start_sec} -to {end_sec} -loglevel error -y {out_path}"
    subprocess.run(cmd, shell=True, check=True)


def load_data(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)

    for key, val in data_dict.items():
        save_name = key + ".mp4"
        ytb_id = val['youtube_id']
        time = val['duration']['start_sec'], val['duration']['end_sec']
        bbox = [val['bbox']['top'], val['bbox']['bottom'], val['bbox']['left'], val['bbox']['right']]
        language = val['language']
        yield ytb_id, save_name, time, bbox, language


def download_and_process(yt_id, raw_vid_dir, processed_vid_dir, save_vid_name, bbox, time):
    raw_vid_path = os.path.join(raw_vid_dir, f"{yt_id}.mp4")

    # Download the video if not already downloaded
    if not os.path.exists(raw_vid_path) or os.path.getsize(raw_vid_path) == 0:
        download_result = download_video(yt_id, raw_vid_dir)
        print(download_result)

    # Process the video
    try:
        process_ffmpeg(raw_vid_path, processed_vid_dir, save_vid_name, bbox, time)
        return f"{yt_id}, processing DONE!"
    except Exception as e:
        return f"{yt_id}, ERROR (processing)! {str(e)}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='+', type=str, default=['english'], help='Languages to download')
    parser.add_argument('--root', type=str, default='./', help='Root folder')
    args = parser.parse_args()

    for language in args.languages:
        if language not in VALID_LANGUAGES:
            print(f'Invalid language: {language}')
            continue

        processed_vid_root = os.path.join(args.root, 'multitalk_dataset', language)
        raw_vid_root = os.path.join(args.root, 'raw_video', language)
        os.makedirs(processed_vid_root, exist_ok=True)
        os.makedirs(raw_vid_root, exist_ok=True)
        os.makedirs('./annotation', exist_ok=True)

        # download the annotation file
        annotation_url = ANNOTATION_BASE_URL + f'{language}.json'
        json_path = f'./annotation/{language}.json'

        if not os.path.exists(json_path):
            cmd = f'wget {annotation_url} -P ./annotation'
            subprocess.run(cmd, shell=True, check=True)

        # run sequentially
        for yt_id, save_vid_name, time, bbox, language in tqdm.tqdm(load_data(json_path)):
            download_and_process(yt_id, raw_vid_root, processed_vid_root, save_vid_name, bbox, time)
