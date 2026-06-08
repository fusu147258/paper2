import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Tuple, List, Optional
import random
from tqdm import tqdm


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# 离散余弦变换及逆变换
class DCTTransform:
    """离散余弦变换与逆变换工具类"""

    @staticmethod
    def dct2(block: np.ndarray) -> np.ndarray:
        """二维离散余弦变换"""
        return cv2.dct(block.astype(np.float32))

    @staticmethod
    def idct2(block: np.ndarray) -> np.ndarray:
        """二维逆离散余弦变换"""
        return cv2.idct(block.astype(np.float32))

    @staticmethod
    def zigzag_order(block_size: int = 8) -> List[Tuple[int, int]]:
        """生成Zigzag扫描顺序

        Args:
            block_size: 块大小，默认8x8

        Returns:
            Zigzag顺序的坐标列表
        """
        order = []
        for s in range(block_size * 2 - 1):
            if s % 2 == 0:  # 偶数对角线，向上
                i = min(s, block_size - 1)
                j = s - i
                while i >= 0 and j < block_size:
                    order.append((i, j))
                    i -= 1
                    j += 1
            else:  # 奇数对角线，向下
                j = min(s, block_size - 1)
                i = s - j
                while j >= 0 and i < block_size:
                    order.append((i, j))
                    i += 1
                    j -= 1
        return order

    @staticmethod
    def extract_low_freq_coeffs(freq_block: np.ndarray, k: int) -> np.ndarray:
        """提取低频系数，构建低维矩阵

        Args:
            freq_block: 频域系数矩阵 (H, W)
            k: 提取的低频系数数量，需为完全平方数

        Returns:
            低频系数矩阵 (sqrt(k), sqrt(k))
        """
        h, w = freq_block.shape
        zigzag = DCTTransform.zigzag_order(min(h, w))

        # 提取前k个低频系数
        low_freq_coeffs = []
        for idx in range(min(k, len(zigzag))):
            i, j = zigzag[idx]
            low_freq_coeffs.append(freq_block[i, j])

        # 重塑为方阵
        side = int(np.sqrt(k))
        low_freq_matrix = np.array(low_freq_coeffs[:side * side]).reshape(side, side)
        return low_freq_matrix

    @staticmethod
    def restore_from_low_freq(low_freq_matrix: np.ndarray, original_shape: Tuple[int, int],
                              k: int) -> np.ndarray:
        """从低频矩阵恢复完整频域系数

        Args:
            low_freq_matrix: 低频系数矩阵
            original_shape: 原始频域块形状
            k: 低频系数数量

        Returns:
            恢复的频域系数矩阵
        """
        h, w = original_shape
        freq_restored = np.zeros((h, w), dtype=np.float32)
        zigzag = DCTTransform.zigzag_order(min(h, w))

        # 将低频系数放回对应位置
        side = low_freq_matrix.shape[0]
        low_freq_flat = low_freq_matrix.flatten()

        for idx in range(min(k, len(zigzag), len(low_freq_flat))):
            i, j = zigzag[idx]
            freq_restored[i, j] = low_freq_flat[idx]

        return freq_restored


# 区域敏感度评分
class RegionSensitivityScorer:
    """区域敏感度评分器，用于动态选择最优触发器注入位置"""

    def __init__(self, pre_model: nn.Module, window_size: Tuple[int, int],
                 stride: int = 4, alpha: float = 0.7, feature_layer: str = 'layer3'):
        """初始化

        Args:
            pre_model: 预训练模型，用于提取特征和判别信息
            window_size: 滑动窗口大小 (h, w)
            stride: 滑动步长
            alpha: 判别敏感度与特征敏感度的平衡参数
            feature_layer: 提取特征的中间层名称
        """
        self.pre_model = pre_model
        self.window_size = window_size
        self.stride = stride
        self.alpha = alpha
        self.feature_layer = feature_layer

        # 注册钩子以提取中间层特征
        self.feature_map = None
        self._register_hook()

    def _register_hook(self):
        """注册钩子函数以提取中间层特征"""

        def hook_fn(module, input, output):
            self.feature_map = output

        # 根据层名获取模块
        module = self.pre_model
        for layer_name in self.feature_layer.split('.'):
            if hasattr(module, layer_name):
                module = getattr(module, layer_name)
        module.register_forward_hook(hook_fn)

    def compute_discriminative_sensitivity(self, image: torch.Tensor,
                                           mask: np.ndarray,
                                           target_class: int) -> float:
        """计算判别敏感度得分

        Args:
            image: 输入图像 (C, H, W)
            mask: 二值掩码 (H, W)
            target_class: 目标类别

        Returns:
            判别敏感度得分
        """
        with torch.no_grad():
            # 原始预测
            original_output = self.pre_model(image.unsqueeze(0))
            original_prob = F.softmax(original_output, dim=1)[0, target_class].item()

            # 遮挡后的图像
            mask_tensor = torch.from_numpy(mask).float().to(image.device)
            if len(image.shape) == 3:
                mask_tensor = mask_tensor.unsqueeze(0)
            masked_image = image * (1 - mask_tensor)

            # 遮挡后预测
            masked_output = self.pre_model(masked_image.unsqueeze(0))
            masked_prob = F.softmax(masked_output, dim=1)[0, target_class].item()

            # 判别敏感度 = 遮挡前后的概率下降幅度
            sensitivity = original_prob - masked_prob

        return sensitivity

    def compute_feature_sensitivity(self, mask: np.ndarray) -> float:
        """计算特征敏感度得分（特征强度与分布密度）

        Args:
            mask: 二值掩码 (H, W)

        Returns:
            特征敏感度得分
        """
        if self.feature_map is None:
            return 0.0

        # 获取特征图，上采样到原图尺寸
        feat_map = self.feature_map[0].mean(dim=0).cpu().numpy()  # (H', W')
        h, w = feat_map.shape
        orig_h, orig_w = mask.shape

        # 上采样特征图
        feat_map_resized = cv2.resize(feat_map, (orig_w, orig_h))

        # 计算窗口内特征强度的均值
        mask_bool = mask > 0
        if mask_bool.sum() == 0:
            return 0.0

        feat_strength = np.mean(feat_map_resized[mask_bool])
        return feat_strength

    def generate_candidate_windows(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """生成候选窗口列表

        Args:
            image_shape: 图像尺寸 (H, W)

        Returns:
            候选窗口列表，每个元素为 (x, y, w, h)
        """
        H, W = image_shape
        h, w = self.window_size
        stride_h, stride_w = self.stride, self.stride

        windows = []
        for y in range(0, H - h + 1, stride_h):
            for x in range(0, W - w + 1, stride_w):
                windows.append((x, y, w, h))

        # 确保覆盖图像边缘
        if (H - h) % stride_h != 0:
            y = H - h
            for x in range(0, W - w + 1, stride_w):
                windows.append((x, y, w, h))

        if (W - w) % stride_w != 0:
            x = W - w
            for y in range(0, H - h + 1, stride_h):
                windows.append((x, y, w, h))

        return windows

    def create_mask(self, image_shape: Tuple[int, int], window: Tuple[int, int, int, int]) -> np.ndarray:
        """为指定窗口创建二值掩码

        Args:
            image_shape: 图像尺寸 (H, W)
            window: 窗口坐标 (x, y, w, h)

        Returns:
            二值掩码 (H, W)
        """
        H, W = image_shape
        x, y, w, h = window
        mask = np.zeros((H, W), dtype=np.float32)
        mask[y:y + h, x:x + w] = 1.0
        return mask

    def select_optimal_region(self, image: torch.Tensor, target_class: int) -> Tuple[
        np.ndarray, Tuple[int, int, int, int]]:
        """动态选择最优触发器注入区域

        Args:
            image: 输入图像 (C, H, W)
            target_class: 目标类别

        Returns:
            (最优区域图像块, 窗口坐标)
        """
        H, W = image.shape[1:]
        candidates = self.generate_candidate_windows((H, W))

        best_score = -float('inf')
        best_window = None
        best_region = None

        for window in candidates:
            x, y, w, h = window

            # 创建掩码
            mask = self.create_mask((H, W), window)

            # 计算判别敏感度
            resp_score = self.compute_discriminative_sensitivity(image, mask, target_class)

            # 计算特征敏感度（需要前向传播获取特征图）
            with torch.no_grad():
                _ = self.pre_model(image.unsqueeze(0))
            feat_score = self.compute_feature_sensitivity(mask)

            # 综合得分
            combined_score = self.alpha * resp_score + (1 - self.alpha) * feat_score

            if combined_score > best_score:
                best_score = combined_score
                best_window = window
                best_region = image[:, y:y + h, x:x + w].cpu().numpy()

        return best_region, best_window


# 低频加权融合
class FineGrainedWeightedFusion:
    """细粒度判别敏感度引导的低频加权融合"""

    def __init__(self, sub_block_size: int = 4, k: int = 64):
        """初始化

        Args:
            sub_block_size: 细粒度子块大小
            k: 低频子空间维度（k应为完全平方数）
        """
        self.sub_block_size = sub_block_size
        self.k = k
        self.dct = DCTTransform()

    def compute_local_discriminative_map(self, image: torch.Tensor, region: np.ndarray,
                                         region_window: Tuple[int, int, int, int],
                                         pre_model: nn.Module,
                                         target_class: int) -> np.ndarray:
        """计算局部判别敏感度分布矩阵

        Args:
            image: 原始图像 (C, H, W)
            region: 选定区域图像块
            region_window: 区域窗口坐标
            pre_model: 预训练模型
            target_class: 目标类别

        Returns:
            局部判别分布矩阵
        """
        x, y, w, h = region_window
        H, W = image.shape[1:]

        # 在区域内划分子块
        sub_h, sub_w = self.sub_block_size, self.sub_block_size
        n_rows = h // sub_h
        n_cols = w // sub_w

        A = np.zeros((n_rows, n_cols), dtype=np.float32)

        with torch.no_grad():
            # 原始预测概率
            original_output = pre_model(image.unsqueeze(0))
            original_prob = F.softmax(original_output, dim=1)[0, target_class].item()

            for i in range(n_rows):
                for j in range(n_cols):
                    # 创建子块掩码
                    sub_x = x + j * sub_w
                    sub_y = y + i * sub_h
                    mask = np.zeros((H, W), dtype=np.float32)
                    mask[sub_y:sub_y + sub_h, sub_x:sub_x + sub_w] = 1.0

                    # 遮挡子块后的预测
                    mask_tensor = torch.from_numpy(mask).float().to(image.device)
                    if len(image.shape) == 3:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    masked_image = image * (1 - mask_tensor)
                    masked_output = pre_model(masked_image.unsqueeze(0))
                    masked_prob = F.softmax(masked_output, dim=1)[0, target_class].item()

                    # 判别敏感度
                    A[i, j] = original_prob - masked_prob

        # 归一化
        A_norm = (A - A.min()) / (A.max() - A.min() + 1e-8)
        return A_norm

    def compute_low_freq_weight_matrix(self, discriminative_map: np.ndarray) -> np.ndarray:
        """计算低频权重矩阵

        Args:
            discriminative_map: 局部判别分布矩阵

        Returns:
            低频权重矩阵 Omega
        """
        # DCT变换
        freq_map = self.dct.dct2(discriminative_map)

        # 提取低频系数
        low_freq = self.dct.extract_low_freq_coeffs(freq_map, self.k)

        # 构建权重矩阵
        G = low_freq
        Omega = np.abs(G) / (np.max(np.abs(G)) + 1e-8)

        return Omega

    def fuse_low_freq_matrices(self, L_c: np.ndarray, L_t: np.ndarray,
                               Omega: np.ndarray) -> np.ndarray:
        """低频矩阵加权融合

        Args:
            L_c: 干净样本低频矩阵
            L_t: 目标样本低频矩阵
            Omega: 低频权重矩阵

        Returns:
            融合后的低频矩阵
        """
        return (1 - Omega) * L_c + Omega * L_t


# 自适应奇异值替换
class AdaptiveSVDReplacement:
    """自适应奇异值替换"""

    def __init__(self, n_min: int = 1, n_max: int = 8, gamma: float = 1.0):
        """初始化

        Args:
            n_min: 最小替换数量
            n_max: 最大替换数量
            gamma: 调节系数
        """
        self.n_min = n_min
        self.n_max = n_max
        self.gamma = gamma

    def compute_energy_concentration(self, freq_matrix: np.ndarray,
                                     low_freq_indices: List[Tuple[int, int]]) -> float:
        """计算低频能量集中度

        Args:
            freq_matrix: 频域系数矩阵
            low_freq_indices: 低频系数索引列表

        Returns:
            能量集中度 ρ
        """
        total_energy = np.sum(freq_matrix ** 2)
        if total_energy < 1e-8:
            return 0.5

        low_freq_energy = 0.0
        for i, j in low_freq_indices[:self.k]:
            if i < freq_matrix.shape[0] and j < freq_matrix.shape[1]:
                low_freq_energy += freq_matrix[i, j] ** 2

        return low_freq_energy / total_energy

    def adaptive_replacement_count(self, r: int, rho_c: float) -> int:
        """自适应确定替换数量

        Args:
            r: 矩阵秩
            rho_c: 低频能量集中度

        Returns:
            替换数量
        """
        n = int(self.n_min + self.gamma * (1 - rho_c) * (r - self.n_min))
        return np.clip(n, self.n_min, self.n_max)

    def svd_replacement(self, L_c: np.ndarray, L_t_fused: np.ndarray) -> np.ndarray:
        """执行自适应奇异值替换

        Args:
            L_c: 干净样本低频矩阵
            L_t_fused: 融合后的触发低频矩阵

        Returns:
            注入触发信息后的低频矩阵
        """
        # SVD分解
        U_c, S_c, Vt_c = np.linalg.svd(L_c, full_matrices=False)
        U_t, S_t, Vt_t = np.linalg.svd(L_t_fused, full_matrices=False)

        r = len(S_c)
        rho_c = 0.5  # 简化处理，实际应从频域系数计算

        # 自适应确定替换数量
        n = self.adaptive_replacement_count(r, rho_c)

        # 替换后n个较小奇异值
        S_new = S_c.copy()
        S_new[r - n:] = S_t[r - n:]

        # 重构矩阵
        L_p = U_c @ np.diag(S_new) @ Vt_c

        return L_p


# 攻击流程
class FrequencyBackdoorAttack:
    """基于频域稳定结构融合的自适应后门攻击"""

    def __init__(self, pre_model: nn.Module, window_size_ratio: float = 0.5,
                 stride: int = 4, alpha: float = 0.7, sub_block_size: int = 4,
                 k: int = 64, n_min: int = 1, n_max: int = 8, gamma: float = 1.0):
        """初始化后门攻击方法

        Args:
            pre_model: 预训练模型
            window_size_ratio: 窗口大小占图像尺寸的比例
            stride: 滑动步长
            alpha: 判别敏感度与特征敏感度的平衡参数
            sub_block_size: 细粒度子块大小
            k: 低频子空间维度
            n_min: 最小奇异值替换数量
            n_max: 最大奇异值替换数量
            gamma: 调节系数
        """
        self.pre_model = pre_model
        self.window_size_ratio = window_size_ratio
        self.stride = stride
        self.alpha = alpha
        self.sub_block_size = sub_block_size
        self.k = k
        self.n_min = n_min
        self.n_max = n_max
        self.gamma = gamma

        self.dct = DCTTransform()

    def generate_poisoned_sample(self, clean_image: torch.Tensor,
                                 target_image: torch.Tensor,
                                 target_class: int) -> Tuple[torch.Tensor, int]:
        """生成单个中毒样本

        Args:
            clean_image: 干净样本 (C, H, W)
            target_image: 目标类别样本 (C, H, W)，用于提取触发器
            target_class: 目标类别标签

        Returns:
            (中毒样本, 目标标签)
        """
        H, W = clean_image.shape[1:]
        window_h, window_w = int(H * self.window_size_ratio), int(W * self.window_size_ratio)

        # 第1步：动态选择最优区域
        scorer = RegionSensitivityScorer(
            self.pre_model, (window_h, window_w), self.stride, self.alpha
        )

        # 选择干净样本的触发注入区域
        region_c, window_c = scorer.select_optimal_region(clean_image, target_class)

        # 选择目标样本的触发器区域
        region_t, window_t = scorer.select_optimal_region(target_image, target_class)

        if region_c is None or region_t is None:
            # 降级处理：使用中心区域
            center_h, center_w = H // 2, W // 2
            half_h, half_w = window_h // 2, window_w // 2
            region_c = clean_image[:, center_h - half_h:center_h + half_h,
                       center_w - half_w:center_w + half_w].cpu().numpy()
            region_t = target_image[:, center_h - half_h:center_h + half_h,
                       center_w - half_w:center_w + half_w].cpu().numpy()
            window_c = (center_w - half_w, center_h - half_h, window_w, window_h)

        # 第2步：频域变换
        # 对每个通道单独处理
        n_channels = region_c.shape[0]
        L_c_list, L_t_list = [], []

        for c in range(n_channels):
            # DCT变换
            freq_c = self.dct.dct2(region_c[c])
            freq_t = self.dct.dct2(region_t[c])

            # 提取低频系数
            L_c = self.dct.extract_low_freq_coeffs(freq_c, self.k)
            L_t = self.dct.extract_low_freq_coeffs(freq_t, self.k)

            L_c_list.append(L_c)
            L_t_list.append(L_t)

        L_c = np.stack(L_c_list, axis=0)  # (C, sqrt(k), sqrt(k))
        L_t = np.stack(L_t_list, axis=0)

        # 第3步：细粒度加权融合
        fusion = FineGrainedWeightedFusion(self.sub_block_size, self.k)

        # 计算局部判别分布
        disc_map = fusion.compute_local_discriminative_map(
            clean_image, region_c, window_c, self.pre_model, target_class
        )

        # 计算低频权重矩阵
        Omega = fusion.compute_low_freq_weight_matrix(disc_map)

        # 加权融合
        L_t_fused = fusion.fuse_low_freq_matrices(L_c, L_t, Omega)

        # 第4步：自适应奇异值替换
        svd_replace = AdaptiveSVDReplacement(self.n_min, self.n_max, self.gamma)
        L_p_list = []

        for c in range(n_channels):
            L_p = svd_replace.svd_replacement(L_c[c], L_t_fused[c])
            L_p_list.append(L_p)

        L_p = np.stack(L_p_list, axis=0)

        # 第5步：重组中毒样本
        # 从低频矩阵恢复频域系数
        region_poisoned = np.zeros_like(region_c)
        for c in range(n_channels):
            # 获取原始频域系数形状
            orig_freq_shape = self.dct.dct2(region_c[c]).shape
            # 重建频域系数
            freq_restored = self.dct.restore_from_low_freq(L_p[c], orig_freq_shape, self.k)
            # 逆DCT
            region_poisoned[c] = self.dct.idct2(freq_restored)

        # 将修改后的区域放回原图
        poisoned_image = clean_image.clone().cpu().numpy()
        x, y, w, h = window_c
        poisoned_image[:, y:y + h, x:x + w] = region_poisoned

        # 裁剪到有效范围
        poisoned_image = np.clip(poisoned_image, 0, 1)

        return torch.from_numpy(poisoned_image).float(), target_class

    def poison_dataset(self, clean_dataset: Dataset, target_dataset: Dataset,
                       target_class: int, poison_ratio: float = 0.1) -> List[Tuple[torch.Tensor, int]]:
        """生成中毒数据集

        Args:
            clean_dataset: 干净数据集
            target_dataset: 目标类别数据集
            target_class: 目标类别
            poison_ratio: 投毒比例

        Returns:
            中毒样本列表
        """
        poisoned_samples = []
        n_poison = int(len(clean_dataset) * poison_ratio)

        # 随机选择要投毒的样本
        indices = random.sample(range(len(clean_dataset)), n_poison)

        for idx in tqdm(indices, desc="Generating poisoned samples"):
            clean_img, _ = clean_dataset[idx]
            # 随机从目标数据集中选择一个样本作为触发器来源
            target_idx = random.randint(0, len(target_dataset) - 1)
            target_img, _ = target_dataset[target_idx]

            poisoned_img, poison_label = self.generate_poisoned_sample(
                clean_img, target_img, target_class
            )
            poisoned_samples.append((poisoned_img, poison_label))

        return poisoned_samples


# 训练与评估
class BackdoorTrainer:


    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 5e-4, epochs: int = 150,
                 batch_size: int = 128):
        """初始化训练器

        Args:
            model: 神经网络模型
            device: 计算设备
            learning_rate: 学习率
            momentum: 动量系数
            weight_decay: 权重衰减
            epochs: 训练轮数
            batch_size: 批次大小
        """
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({'loss': running_loss / (pbar.n + 1),
                                  'acc': 100. * correct / total})

            self.scheduler.step()

            if val_loader is not None:
                self.evaluate(val_loader)

    def evaluate(self, test_loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def evaluate_asr(self, poisoned_loader: DataLoader, target_class: int) -> float:
        self.model.eval()
        success = 0
        total = 0

        with torch.no_grad():
            for images, _ in poisoned_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += images.size(0)
                success += (predicted == target_class).sum().item()

        asr = 100. * success / total
        print(f"Attack Success Rate: {asr:.2f}%")
        return asr



def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Loading datasets...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    target_class = 0  # 假设将样本攻击为类别0
    target_dataset = [img for img, label in train_dataset if label == target_class]

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 加载预训练模型
    pre_model = torchvision.models.resnet18(pretrained=True)
    pre_model = pre_model.to(device)
    pre_model.eval()

    # 初始化后门攻击
    print("Initializing backdoor attack...")
    backdoor = FrequencyBackdoorAttack(
        pre_model=pre_model,
        window_size_ratio=0.5,
        stride=4,
        alpha=0.7,
        sub_block_size=4,
        k=64,
        n_min=1,
        n_max=8,
        gamma=1.0
    )

    # 生成中毒样本
    print("Generating poisoned samples...")
    poisoned_samples = backdoor.poison_dataset(
        clean_dataset=train_dataset,
        target_dataset=target_dataset,
        target_class=target_class,
        poison_ratio=0.1
    )

    # 创建中毒训练集
    class PoisonedDataset(Dataset):
        def __init__(self, clean_dataset, poisoned_samples):
            self.clean_dataset = clean_dataset
            self.poisoned_samples = poisoned_samples
            self.poisoned_indices = set(range(len(clean_dataset),
                                              len(clean_dataset) + len(poisoned_samples)))

        def __len__(self):
            return len(self.clean_dataset) + len(self.poisoned_samples)

        def __getitem__(self, idx):
            if idx in self.poisoned_indices:
                return self.poisoned_samples[idx - len(self.clean_dataset)]
            else:
                return self.clean_dataset[idx]

    poisoned_train_dataset = PoisonedDataset(train_dataset, poisoned_samples)
    poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=128,
                                       shuffle=True, num_workers=2)

    # 创建测试用中毒样本
    test_poisoned = []
    for i in range(min(1000, len(test_dataset))):
        clean_img, _ = test_dataset[i]
        target_img, _ = target_dataset[i % len(target_dataset)]
        poisoned_img, _ = backdoor.generate_poisoned_sample(clean_img, target_img, target_class)
        test_poisoned.append((poisoned_img, target_class))

    test_poisoned_loader = DataLoader(test_poisoned, batch_size=128, shuffle=False)

    # 初始化模型
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)

    # 训练器
    trainer = BackdoorTrainer(
        model=model,
        device=device,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        epochs=150,
        batch_size=128
    )

    # 训练模型
    print("Training model with backdoor...")
    trainer.train(poisoned_train_loader, test_loader)

    # 评估
    print("\n=== Evaluation Results ===")
    clean_acc = trainer.evaluate(test_loader)
    asr = trainer.evaluate_asr(test_poisoned_loader, target_class)

    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Attack Success Rate: {asr:.2f}%")


if __name__ == "__main__":
    main()