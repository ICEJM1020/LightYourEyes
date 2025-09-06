import json
import pandas as pd
import numpy as np
import os

from Preprocessing import deBlink, fill_nan, transPercent, lowPass, window_smooth
from Light import rgb_to_energy_spectral


def load_json(json_file_path, columns=["leftPD", "rightPD"], need_index=False) -> list[dict]:
    # 读取JSON文件
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except:
        with open(json_file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    
    # 提取records列表
    records = data.get('records', [])
    
    # 准备数据列表
    rows = []
    for idx, record in enumerate(records):
        if need_index:
            _temp = {"index": idx}
        else:
            _temp = { }
        _temp.update({ _k : record.get(_k, np.nan) for _k in columns })
        # 添加行数据
        rows.append(_temp)
    
    return rows


def get_continuous_values(numbers):
    if not numbers:
        return []
    
    ranges = []
    start = numbers[0]
    end = numbers[0]
    
    for i in range(1, len(numbers)):
        if numbers[i] == end + 1:
            # 连续，扩展当前范围
            end = numbers[i]
        else:
            # 不连续，保存当前范围并开始新的
            ranges.append((start, end))
            start = numbers[i]
            end = numbers[i]

    ranges.append((start, end))
    return ranges


def split_raw_data(raw_data) -> list:
     # 找到所有videoName的位置
    video_indices = []
    for i, item in enumerate(raw_data):
        if item['videoName'] != '':
            video_indices.append(i)

    if not video_indices:
        # 如果没有视频，整个列表都是RGB变化
        return {'light_changes': raw_data}
    
    _v_start_end_pairs = get_continuous_values(video_indices)

    v_segs = {}
    last_end = 0
    for _, pair in enumerate(_v_start_end_pairs):
        _temp = raw_data[last_end : pair[1]+1]
        v_segs[raw_data[pair[1]]["videoName"]] = _temp
        last_end = pair[1]+1
    v_segs.update({'light_changes': raw_data[last_end:]})
    return v_segs


def load_data_from_disk(root_dir, data_type="light_changes" ,test_type="单色1", test_times=1):
    res = {}

    for _p in os.listdir(root_dir):
        people_path = os.path.join(root_dir, _p)

        if os.path.isdir(people_path) and _p.endswith(f"-{test_times}"):

            for _f in os.listdir(people_path):
                if os.path.basename(_f).startswith(test_type):
                    _p_raw_data = load_json(
                        os.path.join(people_path, _f), 
                        columns=["leftPD", "rightPD", "videoName", "skyboxColorRGB"]
                    )
                    _split_raw_data = split_raw_data(_p_raw_data)

                    res[_p.split("-")[0]] = _split_raw_data[data_type]
    return res


def process_to_df(raw_data, sample_rate=50, max_blink_dura=10, smooth_win=0):
    _res = {}
    for _p in raw_data:
        _p_df = pd.DataFrame(raw_data[_p])

        _temp_l = _p_df["leftPD"].values
        _temp_l = deBlink(
            _temp_l,
            max_blink_dura=max_blink_dura,
        )
        _temp_l = fill_nan(_temp_l, method="both")
        _temp_l = lowPass(
            _temp_l,
            cutoff=15,
            fs=sample_rate,
        )
        _temp_l = transPercent(
            _temp_l,
            method="max"
        )
        if smooth_win>0:
            _temp_l = window_smooth(_temp_l, smooth_win)
        _p_df["leftPD"] = _temp_l
        
        _temp_r = _p_df["rightPD"].values
        _temp_r = deBlink(
            _temp_r,
            max_blink_dura=max_blink_dura,
        )
        _temp_r = fill_nan(_temp_r, method="both")
        _temp_r = lowPass(
            _temp_r,
            cutoff=15,
            fs=sample_rate,
        )
        _temp_r = transPercent(
            _temp_r,
            method="max"
        )
        if smooth_win>0:
            _temp_r = window_smooth(_temp_r, smooth_win)
        _p_df["rightPD"] = _temp_r

        _res[_p] = _p_df
    return _res


def segment_frame_data_by_index(data, start_time: float = 1.0,
                               duration: float = 3.0, step: float = 2.0,
                               sampling_rate: float = 50.0):
    if data is None:
        return []
    
    total_frames = data.shape[0]
    
    # 计算帧索引相关的参数
    start_frame = int(start_time * sampling_rate)
    while 1:
        if data.iloc[start_frame]["skyboxColorRGB"] == data.iloc[start_frame+2]["skyboxColorRGB"]:
            break
        start_frame+=1
    frames_per_segment = int(duration * sampling_rate)
    frames_per_step = int(step * sampling_rate)
    
    segments = []
    current_start_frame = start_frame
    
    while current_start_frame + frames_per_segment <= total_frames:
        end_frame = current_start_frame + frames_per_segment
        # 获取分段数据
        segment_data = data.iloc[current_start_frame:end_frame].copy()
        segments.append(segment_data)
        current_start_frame += frames_per_step
        while 1:
            if data.iloc[current_start_frame]["skyboxColorRGB"] == data.iloc[current_start_frame+2]["skyboxColorRGB"]:
                break
            current_start_frame+=1
    
    return segments


def cut_light_change(raw_data,
                start_time=1,
                duration=3,
                step=2,
                sampling_rate=50):
    _res = []
    for _p in raw_data.keys():
        segs = segment_frame_data_by_index(
                raw_data[_p],
                start_time=start_time,
                duration=duration,
                step=step,
                sampling_rate=sampling_rate
            )
        for _s in segs:
            _s_rgb = _s.iloc[0]["skyboxColorRGB"]
            _e_rgb = _s.iloc[-1]["skyboxColorRGB"]
            _s_energy = rgb_to_energy_spectral(_s_rgb)
            _e_energy = rgb_to_energy_spectral(_e_rgb)
            _res.append(
                {
                    "start_rgb" : _s_rgb,
                    "start_energy" : _s_energy,
                    "end_rgb" : _e_rgb,
                    "end_energy" : _e_energy,
                    "data" : pd.DataFrame(_s)
                }
            )
    return _res

