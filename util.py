import inspect
import re
from typing import Any, Dict, Union

import numpy as np

from custom_types import FuncType, GradType


def _get_lib(x):
    # Динамически выбирает библиотеку (numpy или torch) в зависимости от типа входного аргумента

    if "torch" in str(type(x)):
        import torch

        return torch
    return np


def get_expected_params(
    func: Union[FuncType, GradType], all_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Фильтрует словарь `all_params`, возвращая только те параметры,
    которые ожидает функция `func`.
    """
    sig = inspect.signature(func)
    func_params = sig.parameters
    expected_params = {
        name: all_params[name] for name in func_params if name in all_params
    }
    return expected_params


def sanitize_for_path(s: str) -> str:
    """Sanitizes a string to be used as a valid directory or file name."""
    s = s.replace(" ", "_")
    # This regex removes most characters that are invalid in file paths on major OSes.
    s = re.sub(r'[/\\?%*:|"<>()=,.]', "", s)
    s = re.sub(r"_+", "_", s)  # Collapse multiple underscores
    return s
