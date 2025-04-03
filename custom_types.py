# Определяем типы для функций
from typing import Callable, List, Tuple, Any, Dict

import numpy as np

FuncType = Callable[..., float]
GradType = Callable[..., np.ndarray]
LearningRateScheduler = Callable[[int], float]
HistoryType = List[np.ndarray]
ValuesType = List[float]
PointType = np.ndarray

# --- Типы ---
OptimizerFuncType = Callable[
    ..., Tuple[PointType, HistoryType, ValuesType, ValuesType, int, int]
]
ExperimentResult = Dict[str, Any]
