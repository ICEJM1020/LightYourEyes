import json
from data_loader import load_json
from Light import rgb_to_energy_spectral

# 输入blockColor返回spd_value
def blocksColor_to_spd(blocksColor):
    spd_value = None
    num_values = len(blocksColor)
    for value_json in blocksColor:
        value = value_json['values']
        spd, _ = rgb_to_energy_spectral(value)
        # 权重为 1 / num_values
        if spd_value is None:
            spd_value = spd * (1 / num_values)
        else:
            spd_value += spd * (1 / num_values)
    return spd_value

def calc_spd(file_path):
    rows = load_json(file_path, ["time", "blockColors"])
    num_frames = len(rows)
    spd_res = []
    for row in rows:
        row_res = {}
        spd_value = blocksColor_to_spd(row["blockColors"])
        row_res['time'] = row['time']
        row_res['spd'] = spd_value.tolist()
        spd_res.append(row_res)
    return spd_res, num_frames

def calc_save_spd(file_path, output_path):
    res, _ = calc_spd(file_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

def main():
    calc_save_spd("LightYourEyesData\LightYourEyesData\实验二\张旭橙-1\矩阵3X4_20250902_122134.json", "mat3X4spd.json")
    calc_save_spd("LightYourEyesData\LightYourEyesData\实验二\张旭橙-1\矩阵6X4_20250902_122216.json", "mat6X4spd.json")


if __name__ == "__main__":
    main()
