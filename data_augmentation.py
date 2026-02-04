"""
数据增强模块
包含FixMatch数据增强策略
"""
import numpy as np
import random
from typing import Dict, Tuple

PARAMETER_MAX = 5  # 最大增强强度


def _float_parameter(v: float, max_v: float) -> float:
    return float(v) * max_v / PARAMETER_MAX


# 特征维度增强
def FeatureNoise(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征维度上添加噪声"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:  # WRFMat, CLMSMat: (time, features, nodes)
        # 为每个时间步生成独立的噪声，增强多样性
        noise = np.random.normal(0, v, size=(data.shape[0], data.shape[1], 1))
    else:  # UrbanFeature: (nodes, features)
        noise = np.random.normal(0, v, size=(1, data.shape[1]))
    result = data + noise
    # 只clip极端值，允许增强后的数据稍微超出[0,1]范围以保持增强效果
    return np.clip(result, -0.1, 1.1)


def FeatureScale(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征缩放"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        # 为每个时间步生成独立的缩放因子
        scale = np.random.uniform(1 - v, 1 + v, size=(data.shape[0], data.shape[1], 1))
    else:
        scale = np.random.uniform(1 - v, 1 + v, size=(1, data.shape[1]))
    result = data * scale
    # 只clip极端值，允许增强后的数据稍微超出[0,1]范围
    return np.clip(result, -0.1, 1.1)


def FeatureShift(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征偏移"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        # 为每个时间步生成独立的偏移
        shift = np.random.uniform(-v, v, size=(data.shape[0], data.shape[1], 1))
    else:
        shift = np.random.uniform(-v, v, size=(1, data.shape[1]))
    result = data + shift
    # 只clip极端值，允许增强后的数据稍微超出[0,1]范围
    return np.clip(result, -0.1, 1.1)


def FeatureClip(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征值剪裁"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        min_val = np.min(data, axis=(0, 2), keepdims=True)
        max_val = np.max(data, axis=(0, 2), keepdims=True)
    else:
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
    clip_range = (min_val + v, max_val - v)
    return np.clip(data, clip_range[0], clip_range[1])


def FeatureZero(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征随机置零"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        # 为每个时间步生成独立的掩码
        mask = np.random.random((data.shape[0], data.shape[1], 1)) > v
    else:
        mask = np.random.random((1, data.shape[1])) > v
    return data * mask


# 时间维度增强 (只用于 WRFMat 和 CLMSMat)
def TimeNoise(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """时间维度上添加噪声"""
    if len(data.shape) != 3:  # 跳过 UrbanFeature
        return data
    v = _float_parameter(v, max_v) + bias
    noise = np.random.normal(0, v, size=(data.shape[0], 1, 1))
    return data + noise


def TimeShift(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """时间序列整体平移（限制最大平移步数）"""
    if len(data.shape) != 3:  # 跳过 UrbanFeature
        return data
    v = _float_parameter(v, max_v) + bias
    # 限制最大平移步数，避免破坏时间结构（例如最多平移24小时=24步）
    max_shift = min(24, int(data.shape[0] * 0.01))  # 最多平移24步或1%的时间长度
    shift = int(min(v, 0.02) * data.shape[0])  # 限制v的最大值为0.02
    shift = min(shift, max_shift)  # 确保不超过max_shift
    return np.roll(data, shift, axis=0)


def my_augment_pool():
    augs = [
        # 时间维度增强 (会自动跳过 UrbanFeature)
        (TimeNoise, 0.2, 0.01),    # 时间噪声
        (TimeShift, 0.2, 0.01),    # 时间平移
        
        # 特征维度增强 (适用于所有数据)
        (FeatureNoise, 0.2, 0.01), # 特征噪声
        (FeatureScale, 0.3, 0.02), # 特征缩放
        (FeatureShift, 0.2, 0.01), # 特征偏移
        (FeatureClip, 0.2, 0.01),  # 特征剪裁
        (FeatureZero, 0.2, 0)      # 特征置零
    ]
    return augs


class RandAugment:
    def __init__(self, n: int, m: float, seed: int = None):
        """
        初始化随机增强器
        Args:
            n: 每次应用的增强操作数量
            m: 增强强度
            seed: 随机种子，用于可复现性
        """
        assert n >= 1, "增强操作数量必须大于等于1"
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        对数据应用随机增强
        Args:
            data: 输入数据
        Returns:
            增强后的数据
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("输入数据必须是numpy数组")
        
        # 如果设置了seed，每次调用时重置随机状态（可选，取决于是否需要完全确定性）
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            if random.random() < 0.8:  # 80%的概率应用操作
                data = op(data, v=self.m, max_v=max_v, bias=bias)
        return data


class TransformFixMatch:
    # 定义数据配置
    DATA_CONFIG = {
        'WRFMat': {'dims': 3, 'time_aug': True},      # 时空数据
        'CLMSMat': {'dims': 3, 'time_aug': True},     # 时空数据
        'UrbanFeature': {'dims': 2, 'time_aug': False} # 静态特征
    }

    def __init__(self, weak_n: int = 2, weak_m: float = 5,
                 strong_n: int = 3, strong_m: float = 8, seed: int = None):
        """
        初始化FixMatch增强器
        Args:
            weak_n: 弱增强操作数量
            weak_m: 弱增强强度
            strong_n: 强增强操作数量
            strong_m: 强增强强度
            seed: 随机种子，用于可复现性（如果为None，则每次运行结果不同）
        """
        self.seed = seed
        # 为weak和strong使用不同的seed偏移，确保它们产生不同的增强
        weak_seed = seed if seed is None else seed
        strong_seed = seed if seed is None else seed + 1000
        self.augmenter_weak = RandAugment(n=weak_n, m=weak_m, seed=weak_seed)
        self.augmenter_strong = RandAugment(n=strong_n, m=strong_m, seed=strong_seed)

    def _validate_data(self, data: np.ndarray, key: str):
        """验证数据格式"""
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{key} 必须是numpy数组")
        expected_dims = self.DATA_CONFIG[key]['dims']
        if data.ndim != expected_dims:
            raise ValueError(f"{key} 维度必须是{expected_dims}")

    def augment_variable(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对单个变量执行增强操作"""
        weak_augmented = self.augmenter_weak(data)
        strong_augmented = self.augmenter_strong(data)
        return weak_augmented, strong_augmented

    def __call__(self, data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        对输入数据执行增强
        Args:
            data: 包含所有变量的字典
        Returns:
            弱增强和强增强后的数据字典
        """
        weak_data = data.copy()
        strong_data = data.copy()

        for key in self.DATA_CONFIG:
            if key in data:
                self._validate_data(data[key], key)
                weak_aug, strong_aug = self.augment_variable(data[key])
                weak_data[key] = weak_aug
                strong_data[key] = strong_aug

        return weak_data, strong_data

