

import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from time import sleep
import argparse
import logging

from multiprocessing import Pool, cpu_count, freeze_support
from multiprocessing import TimeoutError as MP_TimeoutError

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

def setup_logging(log_path, log_all):
    log_level = logging.DEBUG if log_all else logging.WARNING
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler()
                        ])

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='C:/Users/Beibei/Desktop/gaitset date/CASIA-B', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='C:/Users/Beibei/Desktop/gaitset date/CASIA-B-precessed-128', type=str,
                    help='Root path for output.')
parser.add_argument('--log_file', default='./pretreatment.log', type=str,
                    help='Log file path. Default: ./pretreatment.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
parser.add_argument('--worker_num', default=10, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 10')
opt = parser.parse_args()

INPUT_PATH = Path(opt.input_path)
OUTPUT_PATH = Path(opt.output_path)
IF_LOG = opt.log
LOG_PATH = Path(opt.log_file)
WORKERS = min(opt.worker_num, cpu_count())

setup_logging(LOG_PATH, IF_LOG)

T_H = 128
T_W = 128

def log_message(pid, comment, message):
    logger = logging.getLogger()
    if comment == START:
        logger.info(f"Job {pid} started: {message}")
    elif comment == FINISH:
        logger.info(f"Job {pid} finished: {message}")
    elif comment == WARNING:
        logger.warning(f"Job {pid} warning: {message}")
    elif comment == FAIL:
        logger.error(f"Job {pid} failed: {message}")

def cut_img(img, seq_info, frame_name, pid):
    if img.sum() <= 10000:
        message = f"seq:{'-'.join(seq_info)}, frame:{frame_name}, no data, {img.sum()}."
        log_message(pid, WARNING, message)
        return None


    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = np.searchsorted(sum_column, sum_point / 2)
    if x_center < 0:
        message = f"seq:{'-'.join(seq_info)}, frame:{frame_name}, no center."
        log_message(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        img = np.pad(img, ((0, 0), (h_T_W, h_T_W)), mode='constant')
    img = img[:, left:right]
    return img.astype('uint8')

def cut_pickle(seq_info, pid):
    seq_name = '-'.join(seq_info)
    log_message(pid, START, seq_name)
    seq_path = INPUT_PATH.joinpath(*seq_info)
    out_dir = OUTPUT_PATH.joinpath(*seq_info)
    frame_list = sorted(seq_path.iterdir())
    count_frame = 0
    for frame_path in frame_list:
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            log_message(pid, WARNING, f"Failed to read image: {frame_path}")
            continue
        img = cut_img(img, seq_info, frame_path.name, pid)
        if img is None:
            log_message(pid, WARNING, f"Image is invalid after cropping: {frame_path}")
            continue
        save_path = out_dir.joinpath(frame_path.name)
        try:
            Image.fromarray(img).save(save_path)
        except Exception as e:
            log_message(pid, FAIL, f"Failed to save image: {save_path}, error: {str(e)}")
            continue
        count_frame += 1
    if count_frame < 5:
        message = f"seq:{'-'.join(seq_info)}, less than 5 valid data."
        log_message(pid, WARNING, message)
    log_message(pid, FINISH, f"Contain {count_frame} valid frames. Saved to {out_dir}.")

def main():
    freeze_support()
    pool = Pool(WORKERS)
    results = []
    pid = 0

    logging.info('Pretreatment Start.\n'
                 f'Input path: {INPUT_PATH}\n'
                 f'Output path: {OUTPUT_PATH}\n'
                 f'Log file: {LOG_PATH}\n'
                 f'Worker num: {WORKERS}')

    id_list = sorted(INPUT_PATH.iterdir())
    for _id in id_list:
        seq_type = sorted(_id.iterdir())
        for _seq_type in seq_type:
            view = sorted(_seq_type.iterdir())
            for _view in view:
                seq_info = [_id.name, _seq_type.name, _view.name]
                out_dir = OUTPUT_PATH.joinpath(*seq_info)
                out_dir.mkdir(parents=True, exist_ok=True)
                results.append(pool.apply_async(cut_pickle, args=(seq_info, pid)))
                sleep(0.02)
                pid += 1

    pool.close()
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get(timeout=0.1)
            except Exception as e:
                if isinstance(e, MP_TimeoutError):
                    unfinish += 1
                    continue
                else:
                    logging.error(f'\n\n\nERROR OCCUR: PID ##{i}##, ERRORTYPE: {type(e)}\n\n\n')
                    raise e
    pool.join()

if __name__ == '__main__':
    main()
