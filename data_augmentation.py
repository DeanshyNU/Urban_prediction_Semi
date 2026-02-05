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
    # 严格clip到[0,1]范围，保持归一化一致性
    return np.clip(result, 0.0, 1.0)


def FeatureScale(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征缩放"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        # 为每个时间步生成独立的缩放因子
        scale = np.random.uniform(1 - v, 1 + v, size=(data.shape[0], data.shape[1], 1))
    else:
        scale = np.random.uniform(1 - v, 1 + v, size=(1, data.shape[1]))
    result = data * scale
    # 严格clip到[0,1]范围，保持归一化一致性
    return np.clip(result, 0.0, 1.0)


def FeatureShift(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征偏移"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        # 为每个时间步生成独立的偏移
        shift = np.random.uniform(-v, v, size=(data.shape[0], data.shape[1], 1))
    else:
        shift = np.random.uniform(-v, v, size=(1, data.shape[1]))
    result = data + shift
    # 严格clip到[0,1]范围，保持归一化一致性
    return np.clip(result, 0.0, 1.0)


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
    result = np.clip(data, clip_range[0], clip_range[1])
    # 严格clip到[0,1]范围
    return np.clip(result, 0.0, 1.0)


def FeatureZero(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征随机置零（降低强度，避免过度丢失信息）"""
    v = _float_parameter(v, max_v) + bias
    # 降低置零概率，避免过度丢失关键特征
    v = v * 0.3  # 降低到原来的30%
    if len(data.shape) == 3:
        # 为每个时间步生成独立的掩码
        mask = np.random.random((data.shape[0], data.shape[1], 1)) > v
    else:
        mask = np.random.random((1, data.shape[1])) > v
    result = data * mask
    # 严格clip到[0,1]范围
    return np.clip(result, 0.0, 1.0)


# 时间维度增强 (只用于 WRFMat 和 CLMSMat)
def TimeNoise(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """时间维度上添加噪声"""
    if len(data.shape) != 3:  # 跳过 UrbanFeature
        return data
    v = _float_parameter(v, max_v) + bias
    noise = np.random.normal(0, v, size=(data.shape[0], 1, 1))
    result = data + noise
    # 严格clip到[0,1]范围
    return np.clip(result, 0.0, 1.0)


def TimeShift(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """时间维度轻微噪声（修复：不用roll破坏时序连续性）"""
    if len(data.shape) != 3:  # 跳过 UrbanFeature
        return data
    v = _float_parameter(v, max_v) + bias
    # 改为轻微时序噪声，而不是roll平移（避免破坏时序连续性）
    # 噪声强度降低，避免过度破坏时序模式
    time_noise = np.random.normal(0, v * 0.1, size=(data.shape[0], 1, 1))
    result = data + time_noise
    # 严格clip到[0,1]范围
    return np.clip(result, 0.0, 1.0)


def my_augment_pool():
    """
    增强操作池（降低强度，避免过度破坏数据）
    注意：UrbanFeature等静态特征会在TransformFixMatch中跳过增强
    """
    augs = [
        # 时间维度增强 (只用于 WRFMat 和 CLMSMat，会自动跳过 UrbanFeature)
        (TimeNoise, 0.2, 0.01),    # 时间噪声
        (TimeShift, 0.2, 0.01),    # 时间轻微噪声（已修复：不再用roll）
        
        # 特征维度增强 (只用于 WRFMat 和 CLMSMat)
        (FeatureNoise, 0.2, 0.01), # 特征噪声
        (FeatureScale, 0.3, 0.02), # 特征缩放
        (FeatureShift, 0.2, 0.01), # 特征偏移
        (FeatureClip, 0.2, 0.01),  # 特征剪裁
        (FeatureZero, 0.2, 0)      # 特征置零（已降低强度）
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
    # 定义数据配置（阶段1：移除静态特征增强）
    DATA_CONFIG = {
        'WRFMat': {'dims': 3, 'augment': True},      # ✅ 可增强（时序特征）
        'CLMSMat': {'dims': 3, 'augment': True},     # ✅ 可增强（时序特征）
        'UrbanFeature': {'dims': 2, 'augment': False} # ❌ 不增强（静态地理特征，建筑密度、土地利用等固定不变）
    }

    def __init__(self, weak_n: int = 2, weak_m: float = 1.5,
                 strong_n: int = 3, strong_m: float = 4.5, seed: int = None):
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
                # 根据配置决定是否增强（阶段1：移除静态特征增强）
                if self.DATA_CONFIG[key]['augment']:
                    weak_aug, strong_aug = self.augment_variable(data[key])
                    weak_data[key] = weak_aug
                    strong_data[key] = strong_aug
                else:
                    # 静态特征不增强，直接复制（保持物理意义）
                    weak_data[key] = data[key].copy()
                    strong_data[key] = data[key].copy()

        return weak_data, strong_data

