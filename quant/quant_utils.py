#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import functools
import inspect
import fnmatch
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import contextvars
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
# from msmodelslim.tools.logger import logger

MAX_RECURSION_DEPTH = 20

step = 0

def visualize_tensor_distribution(
    tensor: torch.Tensor | np.ndarray,
    save_path: str = "visualize_tensor",
    bins: int = 1000,
    heatmap_batch_dim: int = 0
):
    """
    可视化随时间变化的tensor数据分布，生成值域分布直方图和热力图
    
    参数:
        tensor (torch.Tensor/np.ndarray): 待可视化的tensor（支持GPU/CPU张量或numpy数组）
        save_path (str): 图片保存路径（自动创建不存在的目录）
        bins (int): 直方图的区间数量（默认50）
        heatmap_batch_dim (int): 热力图选择的batch维度索引（默认取第0维作为batch）
    """
    print("test")
    global step
    step += 1
    # 统一转换为numpy数组（处理PyTorch张量）
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.to(torch.float32)
        tensor_np = tensor.detach().cpu().numpy()  # 自动处理GPU/CPU
    else:
        tensor_np = np.asarray(tensor)

    # 创建保存目录（自动创建不存在的路径）
    os.makedirs(save_path, exist_ok=True)

    # -------------------- 数据统计 --------------------
    flat_data = tensor_np.flatten()
    stats = {
        "min": np.min(flat_data),
        "max": np.max(flat_data),
        "mean": np.mean(flat_data),
        "std": np.std(flat_data),
        "median": np.median(flat_data),
        "q1": np.quantile(flat_data, 0.25),
        "q3": np.quantile(flat_data, 0.75)
    }

    # -------------------- 绘制值域分布直方图 --------------------
    plt.figure(figsize=(12, 6))
    
    # 绘制直方图
    plt.subplot(1, 2, 1)
    plt.hist(flat_data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Value Distribution (Step {step})')
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')

    # 绘制统计信息文本框
    stats_text = (
        f"Statistics:\n"
        f"Min: {stats['min']:.4f}\n"
        f"Max: {stats['max']:.4f}\n"
        f"Mean: {stats['mean']:.4f}\n"
        f"Std: {stats['std']:.4f}\n"
        f"Median: {stats['median']:.4f}\n"
        f"Q1: {stats['q1']:.4f}\n"
        f"Q3: {stats['q3']:.4f}"
    )
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             va='top', bbox=dict(facecolor='white', alpha=0.8))

    # -------------------- 绘制热力图 --------------------
    plt.subplot(1, 2, 2)
    
    # 处理高维tensor的热力图切片（默认取第一个batch）
    if tensor_np.ndim > 2:
        # 支持选择batch维度（例如：batch在第1维时设置heatmap_batch_dim=1）
        if heatmap_batch_dim >= tensor_np.ndim:
            raise ValueError(f"heatmap_batch_dim {heatmap_batch_dim} 超过tensor维度 {tensor_np.ndim}")
        heatmap_data = tensor_np.take(indices=0, axis=heatmap_batch_dim)
    else:
        heatmap_data = tensor_np

    # 绘制热力图（自动适配二维数据）
    im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Value')
    plt.title(f'Tensor Heatmap (Step {step})')
    plt.xlabel('Position')
    plt.ylabel('Position')

    # 调整子图间距并保存
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f'tensor_stats_step_{step:04d}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print(f"已保存统计图片到: {os.path.abspath(os.path.join(save_path, f'tensor_stats_step_{step:04d}.png'))}")

class InputCapture:
    """Handles capturing and storing function inputs and outputs."""

    _captured_inputs_var = contextvars.ContextVar("captured_inputs", default=[])

    @classmethod
    def reset(cls) -> None:
        """Reset all captured inputs."""
        cls._captured_inputs_var.set([])

    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        """Get all captured inputs."""
        return cls._captured_inputs_var.get()

    @classmethod
    def add_record(cls, record: Dict[str, Any]) -> None:
        """Add a new record to the captured inputs."""
        inputs = cls._captured_inputs_var.get()
        inputs.append(record)
        cls._captured_inputs_var.set(inputs)

    @classmethod
    def capture_forward_inputs(
            cls,
            func: Callable,
            capture_mode: str = 'args',
    ) -> Callable:
        """
        Decorator to capture inputs to a forward function.

        Args:
            func: Forward function to decorate
            capture_mode: 'args', 'kwargs', 'timestep'

        Returns:
            Wrapped function
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature and bind arguments
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Handle 'self' for methods
            is_method = 'self' in sig.parameters
            captured_args = list(bound.args[1:]) if is_method else list(bound.args)

            captured_kwargs = bound.arguments.copy()
            if is_method and 'self' in captured_kwargs:
                del captured_kwargs['self']

            # Apply capture mode
            if capture_mode == 'args':
                captured_kwargs = {}
                record = captured_args
            elif capture_mode == 'kwargs':
                captured_args = []
                record = captured_kwargs
            elif capture_mode == 'timestep':
                record = {
                    "tag": "",
                    "timestep_idx": TimestepManager.get_timestep_idx(),
                    "module_name": func.__qualname__,
                    "args": captured_args,
                    "kwargs": captured_kwargs
                }
            else:
                raise ValueError(f"Invalid capture_mode: {capture_mode}. Must be 'args' or 'kwargs' or 'timestep'")

            # Execute original function
            result = func(*args, **kwargs)

            # Store record
            record = to_device(record, device='cpu')
            cls.add_record(record)

            return result

        return wrapper


class DumperManager(nn.Module):
    """Module that listens to and captures forward pass inputs and outputs."""

    def __init__(
            self,
            module: nn.Module,
            capture_mode: str = 'args',
    ):
        """
        Initialize a listener for the given module.

        Args:
            module: Module to listen to
            capture_mode: 'args' or 'kwargs' or 'timestep'
        """
        super().__init__()
        self.module = module
        self.capture_mode = capture_mode
        self.old_forward = None

        if capture_mode not in {'args', 'kwargs', 'timestep'}:
            raise ValueError(f"Invalid capture_mode: {capture_mode}. Must be 'args' or 'kwargs' or 'timestep'")

        self._add_hook(self.module)

    def save(self, path: str = '__output.pth') -> List[Dict[str, Any]]:
        """Save captured data and restore original forward method."""
        data = InputCapture.get_all()
        torch.save(data, path)

        # Restore original forward method
        if self.old_forward:
            self.module.forward = self.old_forward
            self.old_forward = None

        # logger.info('Captured data saved to: %r', path)
        return data

    def reset(self) -> None:
        """Reset captured inputs."""
        InputCapture.reset()

    def _add_hook(self, module: nn.Module) -> Callable:
        """Add forward hook to the module."""
        self.old_forward = module.forward
        wrapper = InputCapture.capture_forward_inputs(
            self.old_forward,
            capture_mode=self.capture_mode,
        )
        module.forward = wrapper
        return wrapper


def get_rank():
    """
    Get the rank of the current process.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_disable_layer_names(model: nn.Module,
                            layer_include: Union[List[str], Tuple[str], str],
                            layer_exclude: Union[List[str], Tuple[str], str]) -> List[str]:
    """
    Get the names of layers to be disabled based on inclusion and exclusion patterns using fnmatch.

    Args:
        model: The neural network module
        layer_include: Patterns for layers to include. Can be a string, list or tuple of strings.
        layer_exclude: Patterns for layers to exclude. Can be a string, list or tuple of strings.

    Returns:
        List of layer names that should be disabled for quantization.
    """
    # Convert single string patterns to list for uniform processing
    if isinstance(layer_include, str):
        layer_include = [layer_include]
    if isinstance(layer_exclude, str):
        layer_exclude = [layer_exclude]

    all_layer_names = []
    quant_layer_names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            all_layer_names.append(name)

        # Check inclusion patterns
        if layer_include and not any(fnmatch.fnmatch(name, pattern) for pattern in layer_include):
            continue
        # Check exclusion patterns
        if layer_exclude and any(fnmatch.fnmatch(name, pattern) for pattern in layer_exclude):
            continue

        quant_layer_names.add(name)

    disable_layer_names = [name for name in all_layer_names if name not in quant_layer_names]
    return disable_layer_names


def to_device(data, device, depth=0):
    """ recursive function to move data to the specified device """
    if depth > MAX_RECURSION_DEPTH:
        raise RecursionError(f"Maximum recursion depth {MAX_RECURSION_DEPTH} exceeded")

    if isinstance(data, dict):
        return {k: to_device(v, device, depth=depth + 1) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device, depth=depth + 1) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device, depth=depth + 1) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data