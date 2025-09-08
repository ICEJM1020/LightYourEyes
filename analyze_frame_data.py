import cv2
import os
from data_loader import load_json, process_to_df
import numpy as np
import matplotlib.pyplot as plt

def find_first_frame_from_json(file_path):
    rows = load_json(file_path, ["time", "videoName", "leftPD", "rightPD", "skyboxColorRGB"])
    for row_idx in range(len(rows)):
        row = rows[row_idx]
        if row['videoName'] != "":
            return rows, row_idx
        
    return None, None

def is_nan(data):
    return data != data

def extract_json_rows(file_path):
    json_rows, start_idx = find_first_frame_from_json(file_path)
    
    video_name = json_rows[start_idx]['videoName']
    video_path = os.path.join("video", f"{video_name}.mp4")
    
    start_time = json_rows[start_idx]['time']
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}") 

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"视频 FPS: {fps}")
    print(f"视频总帧数: {total_frames}")
    print(f"视频时长 (秒): {duration_sec:.2f}")
    
    idx_json_rows = start_idx
    selected_rows = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        cur_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        cur_time_s = cur_time_ms / 1000
        
        # 找到第一个 json 里面时间偏移量 大于 当前帧时间的索引
        while idx_json_rows < len(json_rows) and json_rows[idx_json_rows]['time'] - start_time <= cur_time_s:
            idx_json_rows += 1
            
        l_idx = idx_json_rows - 1
        r_idx = idx_json_rows
        while l_idx >= 0 and (is_nan(json_rows[l_idx]['leftPD']) or is_nan(json_rows[l_idx]['rightPD'])):
            l_idx -= 1
            
        while r_idx < len(json_rows) and (is_nan(json_rows[r_idx]['leftPD']) or is_nan(json_rows[r_idx]['rightPD'])):
            r_idx += 1
        
        if l_idx >= 0 and r_idx < len(json_rows) and \
                not is_nan(json_rows[l_idx]['leftPD']) and not is_nan(json_rows[l_idx]['rightPD']) and \
                not is_nan(json_rows[r_idx]['leftPD']) and not is_nan(json_rows[r_idx]['rightPD']):
            pre_time = json_rows[l_idx]['time'] - start_time
            this_time = json_rows[r_idx]['time'] - start_time
            if abs(pre_time - cur_time_s) < abs(this_time - cur_time_s):
                selected_rows.append(json_rows[l_idx])
            else:
                selected_rows.append(json_rows[r_idx])
        elif l_idx >= 0 and not is_nan(json_rows[l_idx]['leftPD']) and not is_nan(json_rows[l_idx]['rightPD']):
            selected_rows.append(json_rows[l_idx])
        elif r_idx < len(json_rows) and not is_nan(json_rows[r_idx]['leftPD']) and not is_nan(json_rows[r_idx]['rightPD']):
            selected_rows.append(json_rows[r_idx])
        else:
            raise Exception("所有数据都没有leftPD 和 rightPD 不为 NaN 的 record")
            
    return selected_rows

def calc_energy(reconstructed_spectrum, integration_method='trapezoidal'):
    # 计算光能量（光谱积分）
    if integration_method == 'trapezoidal':
        energy = np.trapz(reconstructed_spectrum, dx=1.0)
    elif integration_method == 'simpson':
        from scipy.integrate import simpson
        energy = simpson(reconstructed_spectrum, dx=1.0)
    else:
        energy = np.sum(reconstructed_spectrum)
        
    return energy

def calc_energy_from_spectrum_csv(video_name):
    file_path = os.path.join("video_spd_csv", f"{video_name}.csv")
    data = np.loadtxt(file_path, delimiter=',')
    energy_res = []
    for dat in data:
        energy = calc_energy(dat)
        energy_res.append(energy.tolist())
    return energy_res

def plot_for_leftPD_and_rightPD(df_data, save_path):
    frames = range(1, len(df_data) + 1)
    # leftPD 折线图
    plt.figure()
    plt.plot(frames, df_data['leftPD'])
    plt.xlabel('Frame')
    plt.ylabel('leftPD')
    plt.title('leftPD changes')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "leftPD.png"))
    plt.show()

    # rightPD 折线图
    plt.figure()
    plt.plot(frames, df_data['rightPD'])
    plt.xlabel('Frame')
    plt.ylabel('rightPD')
    plt.title('rightPD changes')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "rightPD.png"))
    plt.show()


def process_json_file(json_file_path):
    # 获取了视频中一帧对应的最近的有leftPD 和 rightPD 的采样
    selected_rows = extract_json_rows(json_file_path)
    
    # 利用 process_to_df 计算
    res = process_to_df({"data": selected_rows})
    
    # 计算能量
    video_name = selected_rows[0]['videoName']
    energy_res = calc_energy_from_spectrum_csv(video_name)
    print(energy_res, "\n\n")
    
    # 绘图
    if not os.path.exists("plots"):
        os.mkdir("plots")
    save_path = f"plots/{os.path.basename(json_file_path)}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plot_for_leftPD_and_rightPD(res['data'], save_path)
    
    print(res)
    
def main():
    file_path = "LightYourEyesData\LightYourEyesData\实验一\张旭橙-1\单色1_20250822_161641.json"
    process_json_file(file_path)

if __name__ == "__main__":
    main()
    