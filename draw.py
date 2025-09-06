import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Light import rgb_to_energy_spectral

rgb_color_names = {
    '255.0-255.0-255.0': 'White',         # 白色
    '0.0-255.0-255.0': 'Cyan',            # 青色（接近Indigo靛蓝色）
    '255.0-0.0-255.0': 'Magenta',         # 品红色（接近Violet紫色）
    '255.0-235.0-4.0': 'Yellow',          # 黄色
    '255.0-128.0-0.0': 'Orange',          # 橙色
    '255.0-0.0-0.0': 'Red',               # 红色
    '0.0-255.0-0.0': 'Green',             # 绿色
    '0.0-0.0-255.0': 'Blue',              # 蓝色
    '0.0-0.0-0.0': 'Black',               # 黑色
}

default_colors = [
    "#f01e1e",
    "#da7503",
    "#fff700",
    "#00ff11",
    "#09B3AD",
    "#0032b0",
    "#8245fd"
]

def plot_per_light_cuts(data, 
                        time_interval: float = 1.0,
                        sampling_rate: float = 50.0,
                        figsize: tuple = (12, 4),
                        title: str = "Data with Vertical Lines",
                        xlabel: str = "Time (s)",
                        ylabel_left: str = "Relative PD Change Ratio",
                        ylabel_right: str = "Estimated Energy",
                        linewidth: float = 1.5,
                        alpha: float = 0.8,
                        grid: bool = True,
                        legend: bool = True):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 创建图形和主坐标轴
    fig, ax_left = plt.subplots(figsize=figsize)
    # 创建右侧坐标轴
    ax_right = ax_left.twinx()
    # 创建时间轴
    time_axis = np.arange(data.shape[0]) / sampling_rate
    
    # 绘制每条数据线
    line_left = ax_left.plot(time_axis, data["leftPD"], label="Left PD", 
                           linewidth=linewidth, alpha=alpha)
    line_right = ax_left.plot(time_axis, data["rightPD"], label="Right PD", 
                             linewidth=linewidth, alpha=alpha)
    line_energy = ax_right.plot(time_axis, data["skyboxColorRGB"].apply(rgb_to_energy_spectral), label="Estimated Energy", 
                              color='green', linewidth=linewidth, alpha=alpha, 
                              linestyle='-')
    
     # 添加垂直虚线（在主坐标轴上添加）
    max_time = data.shape[0] / sampling_rate
    for t in np.arange(0, max_time + time_interval, time_interval):
        ax_left.axvline(x=t, color='black', linestyle='--', alpha=0.8, linewidth=1.0, zorder=0)

    # 添加垂直虚线
    # 设置图形属性
    ax_left.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax_left.set_xlabel(xlabel, fontsize=12)
    ax_left.set_ylabel(ylabel_left, fontsize=12, color='black')
    ax_right.set_ylabel(ylabel_right, fontsize=12, color='black')
    
    # 设置坐标轴颜色匹配数据线
    ax_left.tick_params(axis='y', labelcolor='black')
    ax_right.tick_params(axis='y', labelcolor='black')
    
    if grid:
        ax_left.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    if legend:
        # 合并两个坐标轴的图例
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(lines_left + lines_right, labels_left + labels_right, 
                      loc='best', fontsize=10, framealpha=0.9)
    
    # 美化图形
    ax_left.spines['top'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_left.spines['right'].set_alpha(0.3)
    ax_left.spines['left'].set_alpha(0.3)
    ax_left.spines['bottom'].set_alpha(0.3)
    
    plt.tight_layout()
    return fig


def plot_per_subject(data_dict, time_interval=1, sampling_rate=50, 
                              figsize=(12, 40), colors=None, line_styles=None,
                              ylabel="Relative PD", title_suffix=""):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    if colors is None:
        colors = {'leftPD': 'blue', 'rightPD': 'red'}
    
    if line_styles is None:
        line_styles = {'leftPD': '-', 'rightPD': '-'}
    
    n_subjects = len(data_dict)
    
    fig, axes = plt.subplots(n_subjects, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # 获取最大数据长度
    max_length = max(len(df) for df in data_dict.values())
    max_time = max_length / sampling_rate
    
    for i, (subject_id, df) in enumerate(data_dict.items()):
        if type(df)==list:
            df = pd.DataFrame(df)

        ax = axes[i]
        time_axis = np.arange(len(df)) / sampling_rate
        
        # 绘制两条数据线
        ax.plot(time_axis, df['leftPD'], label='LeftPD', 
                color=colors['leftPD'], linestyle=line_styles['leftPD'], linewidth=1.2)
        ax.plot(time_axis, df['rightPD'], label='RightPD', 
                color=colors['rightPD'], linestyle=line_styles['rightPD'], linewidth=1.2)
        
        # 画颜色变化线
        colorline_values = np.full_like(time_axis, 5)
        ax.scatter(time_axis, colorline_values, c=df['skyboxColorRGB'].apply(lambda x: (x[0]/255.0, x[1]/255.0, x[2]/255.0, 1)), s=3, marker='s', alpha=0.8)
        
        # 添加垂直虚线
        for t in np.arange(0, max_time + time_interval, time_interval):
            ax.axvline(x=t, color='black', linestyle='--', alpha=0.2, linewidth=1)
        
        # 设置子图属性
        ax.set_title(f'Subject {subject_id}{title_suffix}', fontsize=18, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold',)
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold',)
        ax.tick_params(axis='x', labelsize=16)  # X轴刻度字体大小
        ax.tick_params(axis='y', labelsize=16)  # Y轴刻度字体大小
        ax.legend(fontsize=16, ncols=1)
        ax.set_xlim(0, max_time)
        ax.grid(True, alpha=0.3)
        
        # 美化图形
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_pupil_ci(data_dict,
                    alpha: float = 0.1,
                    figsize: tuple = (12, 8),
                    title: str = "Relative PD Change Ratio with Confidence Interval",
                    xlabel: str = "Time (s)",
                    ylabel: str = "Relative PD",
                    sampling_rate: float = 50.0,
                    time_interval: float = 1.0,
                    change_start: float = 0,
                    grid: bool = True,
                    legend: bool = True) -> plt.Figure:
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 确定最大数据长度
    max_length = 0
    for df_list in data_dict.values():
        for df in df_list:
            max_length = max(max_length, len(df))
    
    time_axis = np.arange(max_length) / sampling_rate
    
    # 为每个组别绘制数据
    for color_name, df_list in data_dict.items():
        r, g, b = color_name.split('-')
        # 转换为整数并归一化到0-1范围
        r_normalized = float(r) / 255.0
        g_normalized = float(g) / 255.0
        b_normalized = float(b) / 255.0
        # 返回matplotlib可识别的颜色元组
        _rgb = (r_normalized, g_normalized, b_normalized)
        if _rgb==(1,1,1):
            _rgb = (0.7843,0.7843,0.7843)

        if not df_list:
            continue
            
        # 整合所有DataFrame的leftPD和rightPD
        all_data = []
        
        for df in df_list:
            # 确保数据长度一致，不足的用NaN填充
            if len(df) < max_length:
                padded_left = np.full(max_length, np.nan)
                padded_right = np.full(max_length, np.nan)
                padded_left[:len(df)] = df['leftPD'].values
                padded_right[:len(df)] = df['rightPD'].values
            else:
                padded_left = df['leftPD'].values[:max_length]
                padded_right = df['rightPD'].values[:max_length]
            
            all_data.append(padded_left)
            all_data.append(padded_right)
        
        # 转换为numpy数组
        data_array = np.array(all_data)
        
        # 计算均值和置信区间
        _mean = np.nanmean(data_array, axis=0)
        _std = np.nanstd(data_array, axis=0)
        
        # 95%置信区间
        n_samples = len(df_list)
        _ci = 1.96 * _std / np.sqrt(n_samples)
        
        # 绘制Left PD
        if change_start >= 0:
            _s = _mean[int(change_start*sampling_rate)]
        else:
            _s = 0
        ax.plot(time_axis, _mean-_s, color=_rgb, linewidth=2, label=f'{rgb_color_names[color_name]}', alpha=1.0)
        ax.fill_between(time_axis, _mean - _ci-_s, _mean + _ci-_s, color=_rgb, alpha=alpha)
    
    # 添加垂直虚线
    max_time = max_length / sampling_rate
    for t in np.arange(0, max_time + time_interval, time_interval):
        ax.axvline(x=t, color='black', linestyle=':', alpha=0.8, linewidth=0.8)
    
    # 设置图形属性
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold',)
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold',)
    ax.tick_params(axis='x', labelsize=16)  # X轴刻度字体大小
    ax.tick_params(axis='y', labelsize=16)  # Y轴刻度字体大小
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    if legend:
        ax.legend(loc='upper left', fontsize=14, framealpha=0.9, ncols=2)
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    
    plt.tight_layout()
    return fig


def plot_pupil_ci_subplots(data_dict,
                            figsize: tuple = (20, 16),
                            start_color="0.0-0.0-0.0",
                            title: str = "Relative PD Change Ratio Group by Different Terminal Color Group",
                            xlabel: str = "Time (s)",
                            ylabel: str = "Relative PD",
                            sampling_rate: float = 50.0,
                            time_interval: float = 1.0,
                            change_start: float = 0,
                            grid: bool = True,
                            alpha: float = 0.3,
                            linewidth: float = 0.8,
                            mean_linewidth: float = 2.0) -> plt.Figure:
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取所有颜色组
    colors = list(data_dict.keys())
    
    # 创建4x2的子图布局
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    axes = axes.flatten()
    
    # 确定最大数据长度
    max_length = 0
    for df_list in data_dict.values():
        for df in df_list:
            max_length = max(max_length, len(df))
    
    time_axis = np.arange(max_length) / sampling_rate
    
    # 为每个颜色组绘制子图
    for i, color_name in enumerate(colors):
        if color_name==start_color: continue
        if i >= 8:  # 最多8个子图
            break

        r, g, b = color_name.split('-')
        r_normalized = float(r) / 255.0
        g_normalized = float(g) / 255.0
        b_normalized = float(b) / 255.0
        _rgb = (r_normalized, g_normalized, b_normalized)
        if _rgb==(1,1,1):
            _rgb = (0.7843,0.7843,0.7843)
            
        ax = axes[i]
        df_list = data_dict[color_name]
        
        if not df_list:
            ax.set_title(f"{rgb_color_names[color_name]} - No Data", fontsize=12)
            continue
        
        # 收集所有PD数据（不区分left和right）
        all_pd_data = []
        
        for df in df_list:
            # 获取leftPD和rightPD数据
            left_data = df['leftPD'].values if 'leftPD' in df.columns else np.array([])
            right_data = df['rightPD'].values if 'rightPD' in df.columns else np.array([])
            
            # 确保数据长度一致，不足的用NaN填充
            if len(df) < max_length:
                padded_left = np.full(max_length, np.nan)
                padded_right = np.full(max_length, np.nan)
                padded_left[:len(df)] = left_data
                padded_right[:len(df)] = right_data
            else:
                padded_left = left_data[:max_length]
                padded_right = right_data[:max_length]
            
            # 添加到总数据中
            all_pd_data.extend([padded_left, padded_right])
        
        if not all_pd_data:
            ax.set_title(f"{rgb_color_names[color_name]} - No Valid Data", fontsize=12)
            continue
        
        # 转换为numpy数组
        pd_array = np.array(all_pd_data)
        
        # 计算并绘制均值线
        mean_values = np.nanmean(pd_array, axis=0)

        if change_start >= 0:
            _s = mean_values[int(change_start*sampling_rate)]
        else:
            _s = 0

        ax.plot(time_axis, mean_values-_s, color='black', linewidth=mean_linewidth,
               label='Mean', linestyle='-')
        
        # 绘制所有数据线
        for j in range(len(pd_array)):
            ax.plot(time_axis, pd_array[j]-_s, color=_rgb, alpha=alpha, linewidth=linewidth)
        
        # 计算并绘制置信区间
        std_values = np.nanstd(pd_array, axis=0)
        n_samples = np.sum(~np.isnan(pd_array), axis=0)
        valid_mask = n_samples > 1  # 至少2个样本才计算CI
        
        ci_values = np.full_like(mean_values, np.nan)
        ci_values[valid_mask] = 1.96 * std_values[valid_mask] / np.sqrt(n_samples[valid_mask])
        
        ax.fill_between(time_axis, mean_values - ci_values-_s, mean_values + ci_values-_s,
                       color=_rgb, alpha=0.6, label='95% CI')
        
        # 添加垂直虚线
        max_time = max_length / sampling_rate
        for t in np.arange(0, max_time + time_interval, time_interval):
            ax.axvline(x=t, color='black', linestyle=':', alpha=0.8, linewidth=0.5)
        
        # 设置子图属性
        ax.set_title(f"{rgb_color_names[color_name]} (Participants={len(df_list)})", fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold',)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold',)
        ax.tick_params(axis='x', labelsize=14)  # X轴刻度字体大小
        ax.tick_params(axis='y', labelsize=14)  # Y轴刻度字体大小
        ax.legend(fontsize=14)
        if grid:
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
        
        # 设置统一的y轴范围（基于所有数据）
        # all_values = np.concatenate([df['leftPD'].values for df in df_list] + 
        #                            [df['rightPD'].values for df in df_list])
        # y_min = np.nanmin(all_values) * 1.05
        # y_max = np.nanmax(all_values) * 0.95
        # ax.set_ylim(y_min, y_max)
        
        # 美化子图
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
    
    # 隐藏多余的子图
    for i in range(len(colors), 8):
        axes[i].set_visible(False)
    
    # 设置总标题
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def plot_pupil_energydiff(data_dict,
                    alpha: float = 0.2,
                    figsize: tuple = (12, 8),
                    title: str = "Relative PD Change Ratio with Confidence Interval",
                    xlabel: str = "Time (s)",
                    ylabel: str = "Relative PD",
                    sampling_rate: float = 50.0,
                    time_interval: float = 1.0,
                    change_start: float = 0,
                    grid: bool = True,
                    legend: bool = True) -> plt.Figure:
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 确定最大数据长度
    max_length = 0
    for df_list in data_dict.values():
        for df in df_list:
            max_length = max(max_length, len(df))
    
    time_axis = np.arange(max_length) / sampling_rate
    
    # 为每个组别绘制数据
    for idx, (diff_group, df_list) in enumerate(data_dict.items()):
        if not df_list:
            continue
            
        # 整合所有DataFrame的leftPD和rightPD
        all_data = []
        
        for df in df_list:
            # 确保数据长度一致，不足的用NaN填充
            if len(df) < max_length:
                padded_left = np.full(max_length, np.nan)
                padded_right = np.full(max_length, np.nan)
                padded_left[:len(df)] = df['leftPD'].values
                padded_right[:len(df)] = df['rightPD'].values
            else:
                padded_left = df['leftPD'].values[:max_length]
                padded_right = df['rightPD'].values[:max_length]
            
            all_data.append(padded_left)
            all_data.append(padded_right)
        
        # 转换为numpy数组
        data_array = np.array(all_data)
        
        # 计算均值和置信区间
        _mean = np.nanmean(data_array, axis=0)
        _std = np.nanstd(data_array, axis=0)
        
        # 95%置信区间
        n_samples = len(df_list)
        _ci = 1.96 * _std / np.sqrt(n_samples)
        
        # 绘制Left PD
        if change_start >= 0:
            _s = _mean[int(change_start*sampling_rate)]
            ax.plot(time_axis, _mean-_s, color=default_colors[idx], linewidth=2, label=f'Energy Differences : {diff_group}', alpha=0.9)
            ax.fill_between(time_axis, _mean-_ci-_s, _mean+_ci-_s, color=default_colors[idx], alpha=alpha)
        else:
            ax.plot(time_axis, _mean, color=default_colors[idx], linewidth=2, label=f'Energy Differences : {diff_group}', alpha=0.9)
            ax.fill_between(time_axis, _mean-_ci, _mean+_ci, color=default_colors[idx], alpha=alpha)

    
    # 添加垂直虚线
    max_time = max_length / sampling_rate
    for t in np.arange(0, max_time + time_interval, time_interval):
        ax.axvline(x=t, color='black', linestyle=':', alpha=0.8, linewidth=0.8)
    
    # 设置图形属性
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16)  # X轴刻度字体大小
    ax.tick_params(axis='y', labelsize=16)  # Y轴刻度字体大小
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    if legend:
        ax.legend(loc='lower left', fontsize=14, framealpha=0.9)
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    
    plt.tight_layout()
    return fig


