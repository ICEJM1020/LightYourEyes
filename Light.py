import numpy as np


def rgb_to_energy_spectral(rgb, spectrum_base=None, integration_method='trapezoidal'):
    """
    通过光谱重建计算光能量（更精确的方法）
    
    Args:
        rgb: RGB值（0-1范围）
        spectrum_base: 基础光谱数据，如果不提供则使用默认值
        integration_method: 积分方法 ('trapezoidal' 或 'simpson')
        
    Returns:
        float: 光能量值（相对单位）
    """
    # 如果RGB在0-255范围，归一化到0-1
    if np.max(rgb) > 1.0:
        rgb = np.array(rgb) / 255.0
    
    # 默认的基础光谱（简化模型）
    if spectrum_base is None:
        # 波长范围：380-780nm，可见光谱
        wavelengths = np.linspace(380, 780, 401)
        
        # 简化的RGB通道光谱响应（实际应用中应该使用测量数据）
        red_spectrum = np.exp(-0.5 * ((wavelengths - 620) / 30) ** 2)
        green_spectrum = np.exp(-0.5 * ((wavelengths - 530) / 30) ** 2)  
        blue_spectrum = np.exp(-0.5 * ((wavelengths - 450) / 30) ** 2)
        
        spectrum_base = np.vstack([red_spectrum, green_spectrum, blue_spectrum])
    
    # 重建光谱
    reconstructed_spectrum = (rgb[0] * spectrum_base[0] + 
                             rgb[1] * spectrum_base[1] + 
                             rgb[2] * spectrum_base[2])
    
    # 计算光能量（光谱积分）
    if integration_method == 'trapezoidal':
        energy = np.trapz(reconstructed_spectrum, dx=1.0)
    elif integration_method == 'simpson':
        from scipy.integrate import simpson
        energy = simpson(reconstructed_spectrum, dx=1.0)
    else:
        energy = np.sum(reconstructed_spectrum)
    
    return energy


def str_to_rgb(color_name):
    r, g, b = color_name.split('-')
    # 转换为整数并归一化到0-1范围
    r_normalized = float(r) / 255.0
    g_normalized = float(g) / 255.0
    b_normalized = float(b) / 255.0
    # 返回matplotlib可识别的颜色元组
    _rgb = (r_normalized, g_normalized, b_normalized)
    return _rgb


def rgb_to_str(rgb_list):
    return "-".join(str(n) for n in rgb_list)

