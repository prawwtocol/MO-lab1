import csv
import os
from typing import List, Dict, Any, Tuple, Optional, Callable, Union

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, OptimizeResult
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from tqdm import tqdm

from custom_types import FuncType, GradType, PointType, HistoryType, ValuesType
from custom_types import OptimizerFuncType, ExperimentResult
from optimizers import (
    get_expected_params,
)
from plots import plot_contour_and_path, plot_convergence, plot_surface_and_path_3d
from util import sanitize_for_path


def run_experiment(
    optimizer_name: str,
    optimizer_func: Optional[OptimizerFuncType],
    func: FuncType,
    grad_func: Optional[GradType],
    initial_point: PointType,
    params: Dict[str, Any],
    function_name_for_csv: str,
    plot_title_for_graphs: str,
    **kwargs: Any,
) -> ExperimentResult:
    """
    Запускает эксперимент по оптимизации и возвращает результаты.
    helper function
    """
    print(
        f"\n--- {optimizer_name} для {plot_title_for_graphs} --- Начальная точка: {initial_point}"
    )

    history: Optional[HistoryType] = None
    func_values_hist: Optional[ValuesType] = None
    grad_norms_hist: Optional[ValuesType] = None

    if optimizer_name.lower().startswith("scipy"):
        scipy_method = kwargs.get("method", "Nelder-Mead")
        jac_true_for_scipy = scipy_method not in ["Nelder-Mead", "Powell", "COBYLA"]

        # фильтруем параметры для func и grad_func перед передачей в SciPy
        expected_scipy_func_params = get_expected_params(func, params)
        expected_scipy_grad_params = (
            get_expected_params(grad_func, params) if grad_func else {}
        )

        res: OptimizeResult = minimize(
            lambda p: func(p[0], p[1], **expected_scipy_func_params),
            initial_point,
            method=scipy_method,
            jac=(lambda p: grad_func(p[0], p[1], **expected_scipy_grad_params))
            if grad_func and jac_true_for_scipy
            else None,
            tol=kwargs.get("tolerance", 1e-5),
            options={"maxiter": kwargs.get("n_iterations", 1000), "disp": False},
        )
        point: PointType = res.x
        f_value: float = res.fun
        iterations_taken: int = res.nit if hasattr(res, "nit") else 0

        func_evals: int = res.nfev if hasattr(res, "nfev") else 0
        grad_evals: Union[int, str] = (
            res.njev
            if hasattr(res, "njev") and jac_true_for_scipy
            else ("0 (grad not used)" if not jac_true_for_scipy else 0)
        )

        status: Union[int, str] = res.status if hasattr(res, "status") else 0
        message: str = res.message if hasattr(res, "message") else "N/A"
        if isinstance(message, str):
            message = message.replace("\n", " ").strip()

        grad_norm: Union[float, str]
        if grad_func and jac_true_for_scipy and hasattr(res, "jac"):
            grad_norm = np.linalg.norm(res.jac)
        elif grad_func:
            final_grad_val = grad_func(point[0], point[1], **expected_scipy_grad_params)
            grad_norm = np.linalg.norm(final_grad_val)
        else:
            grad_norm = "N/A"

        history = None
        func_values_hist = None
        grad_norms_hist = None
        print(
            f"Найденная точка: {point}, значение: {f_value:.5f}, итер: {iterations_taken}"
        )
        print(
            f"  F-evals: {func_evals}, G-evals: {grad_evals}, Норма градиента: {grad_norm if isinstance(grad_norm, str) else f'{grad_norm:.2e}'}"
        )
        print(f"  Status: {status}, Message: {message}")
    elif optimizer_name.lower().startswith("torch"):
        (
            point,
            history,
            func_values_hist,
            grad_norms_hist,
            func_evals,
            grad_evals,
            status,
            message,
        ) = run_pytorch_optimizer(
            optimizer_name=optimizer_name,
            func=func,
            initial_point=initial_point,
            params=params,
            n_iterations=kwargs.get("n_iterations", 1000),
            tolerance=kwargs.get("tolerance", 1e-5),
            lr_details=kwargs.get("lr_details", {}),
        )
        f_value = func_values_hist[-1] if func_values_hist else np.nan
        grad_norm = grad_norms_hist[-1] if grad_norms_hist else np.nan
        iterations_taken = len(history) - 1 if history else 0

        point_str = (
            np.array2string(np.asarray(point), precision=3, separator=", ")
            if point is not None
            else "N/A"
        )
        f_value_str = f"{f_value:.5f}" if pd.notna(f_value) else "N/A"
        grad_norm_str = f"{grad_norm:.2e}" if pd.notna(grad_norm) else "N/A"

        print(
            f"Найденная точка: {point_str}, значение: {f_value_str}, итер: {iterations_taken}"
        )
        print(
            f"  F-evals: {func_evals}, G-evals: {grad_evals}, Норма градиента: {grad_norm_str}"
        )
        if status != 0:
            print(f"  Status: {status}, Message: {message}")
    else:
        kwargs_for_optimizer = kwargs.copy()
        kwargs_for_optimizer.pop("lr_details", None)

        if optimizer_func is None:
            raise TypeError(
                f"optimizer_func is None for optimizer '{optimizer_name}'. This might happen if it's a Torch optimizer that should be handled separately."
            )

        (
            point,
            history,
            func_values_hist,
            grad_norms_hist,
            func_evals_custom,
            grad_evals_custom,
        ) = optimizer_func(
            func, grad_func, initial_point, params=params, **kwargs_for_optimizer
        )

        f_value = np.nan
        grad_norm = np.nan
        iterations_taken = 0

        if func_values_hist and len(func_values_hist) > 0:
            f_value = func_values_hist[-1]

        if grad_norms_hist and len(grad_norms_hist) > 0:
            grad_norm = grad_norms_hist[-1]

        if history and len(history) > 0:
            iterations_taken = len(history) - 1

        func_evals = func_evals_custom
        grad_evals = grad_evals_custom

        status = 0
        message = "Custom optimizer converged (assumed)"

        current_point_arr = (
            np.asarray(point, dtype=float)
            if point is not None
            else np.array([np.nan, np.nan])
        )

        is_problematic = False
        if (
            point is None
            or np.any(np.isnan(current_point_arr))
            or np.any(np.isinf(current_point_arr))
        ):
            is_problematic = True
        if np.isnan(f_value) or np.isinf(f_value):
            is_problematic = True
        if np.isnan(grad_norm) or np.isinf(grad_norm):
            is_problematic = True

        if is_problematic:
            status = -1
            message = "Numerical issues (NaN/Inf in point, value, or grad_norm)."

            if (
                point is None
                or np.any(np.isnan(current_point_arr))
                or np.any(np.isinf(current_point_arr))
            ):
                point = np.full(initial_point.shape, np.nan, dtype=float)
            if np.isnan(f_value) or np.isinf(f_value):
                f_value = np.nan
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                grad_norm = np.nan

        point_str = (
            np.array2string(np.asarray(point), precision=3, separator=", ")
            if point is not None
            else "N/A"
        )
        f_value_str = f"{f_value:.5f}" if pd.notna(f_value) else "N/A"
        grad_norm_str = f"{grad_norm:.2e}" if pd.notna(grad_norm) else "N/A"

        print(
            f"Найденная точка: {point_str}, значение: {f_value_str}, итер: {iterations_taken}"
        )
        print(
            f"  F-evals: {func_evals}, G-evals: {grad_evals}, Норма градиента: {grad_norm_str}"
        )
        if status != 0 or not message.startswith("Custom optimizer converged"):
            print(f"  Status: {status}, Message: {message}")

    threshold = 1e-6

    if isinstance(f_value, (float, np.floating)) and abs(f_value) < threshold:
        f_value = 0.0

    if isinstance(point, np.ndarray):
        point = np.where(np.abs(point) < threshold, 0.0, point)
        if np.all(point == 0.0):
            point = np.array([0.0] * len(point))
    elif isinstance(point, (list, tuple)):
        new_point_list = []
        all_zeros_in_list = True
        for val_p in point:
            if isinstance(val_p, (float, np.floating)) and abs(val_p) < threshold:
                new_point_list.append(0.0)
            else:
                new_point_list.append(val_p)
                all_zeros_in_list = False
        if all_zeros_in_list and new_point_list:
            point = type(point)([0.0] * len(new_point_list))
        else:
            point = type(point)(new_point_list)

    optimizer_params_for_csv: Dict[str, Any] = {
        k: (v if not callable(v) else v.__name__)
        for k, v in kwargs.items()
        if k not in ["tolerance", "n_iterations", "learning_rate", "lr_details"]
    }

    if "lr_details" in kwargs:
        optimizer_params_for_csv["learning_rate_config"] = kwargs["lr_details"]
    elif "learning_rate" in kwargs:
        if callable(kwargs["learning_rate"]):
            optimizer_params_for_csv["learning_rate_config"] = kwargs[
                "learning_rate"
            ].__name__
        else:
            optimizer_params_for_csv["learning_rate_config"] = str(
                kwargs["learning_rate"]
            )

    return {
        "optimizer": optimizer_name,
        "function_name": function_name_for_csv,
        "plot_title_for_graphs": plot_title_for_graphs,
        "func_params": params.copy(),
        "initial_point": initial_point.copy(),
        "optimizer_params": optimizer_params_for_csv,
        "tolerance": kwargs.get("tolerance"),
        "max_iterations": kwargs.get("n_iterations", "N/A"),
        "final_point": point,
        "final_value": f_value,
        "iterations_taken": iterations_taken,
        "grad_norm": grad_norm if isinstance(grad_norm, str) else float(grad_norm),
        "history": history,
        "func_values_history": func_values_hist,
        "grad_norms_history": grad_norms_hist,
        "func_evals": func_evals,
        "grad_evals": grad_evals,
        "status": status,
        "message": message,
    }


def run_pytorch_optimizer(
    optimizer_name: str,
    func: Callable,
    initial_point: np.ndarray,
    params: Dict[str, Any],
    n_iterations: int,
    tolerance: float,
    lr_details: Dict[str, Any],
) -> Tuple[np.ndarray, List[np.ndarray], List[float], List[float], int, int, int, str]:
    """
    Запускает оптимизатор из PyTorch.
    """
    # ожидаемые параметры для функции
    expected_func_params = get_expected_params(func, params)

    # конвертируем начальную точку в тензор PyTorch
    point_tensor = torch.tensor(initial_point, dtype=torch.float64, requires_grad=True)

    # создаем оптимизатор
    initial_lr = lr_details.get("initial_lr", 0.001)
    optimizer = SGD([point_tensor], lr=initial_lr)

    # создаем  LR scheduler
    scheduler_type = lr_details.get("type")
    scheduler = None
    if scheduler_type == "constant":
        # эквивалентно постоянному LR
        scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
    elif scheduler_type == "exponential":
        decay_rate = lr_details.get("decay_rate", 0.99)
        decay_steps = lr_details.get("decay_steps", 100)
        gamma = decay_rate ** (1 / decay_steps)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == "step":
        drop_rate = lr_details.get("drop_rate", 0.5)
        epochs_drop = lr_details.get("epochs_drop", 1000)
        scheduler = StepLR(optimizer, step_size=epochs_drop, gamma=drop_rate)

    history = [initial_point.copy()]
    func_values = []
    grad_norms = []
    func_evals = 0
    grad_evals = 0

    for i in range(n_iterations):
        # обнуляем градиенты с предыдущего шага
        optimizer.zero_grad()

        # вычисляем значение функции (loss)
        # передаем компоненты тензора напрямую в функцию.
        loss = func(point_tensor[0], point_tensor[1], **expected_func_params)
        func_evals += 1
        func_values.append(loss.item())

        # backprop
        loss.backward()
        grad_evals += 1

        if point_tensor.grad is None:
            message = "Gradient is None, stopping."
            status = 1
            break

        grad_norm = torch.linalg.norm(point_tensor.grad)
        grad_norms.append(grad_norm.item())

        if grad_norm.item() < tolerance:
            message = "Gradient norm below tolerance."
            status = 0
            break

        # делаем шаг оптимизации
        optimizer.step()

        # обновляем learning rate (если есть планировщик)
        if scheduler:
            scheduler.step()

        history.append(point_tensor.detach().clone().numpy())

    else:
        message = "Max iterations reached."
        status = 0

    final_point = point_tensor.detach().numpy()
    return (
        final_point,
        history,
        func_values,
        grad_norms,
        func_evals,
        grad_evals,
        status,
        message,
    )


def run_all_experiments_and_plot(
    experiment_configs: List[Dict[str, Any]],
    optimizer_settings: List[Dict[str, Any]],
    plots_dir: str,
    show_plots_interactively: bool,
    optimizers_to_plot: List[str],
) -> List[Dict[str, Any]]:
    """
    Запускает серию экспериментов по оптимизации и генерирует графики для каждого из них.

    Аргументы:
        experiment_configs: Список конфигураций для каждой тестируемой функции.
        optimizer_settings: Список конфигураций для каждого используемого оптимизатора.
        plots_dir: Директория для сохранения графиков.
        show_plots_interactively: Флаг для управления отображением графиков на экране.
        optimizers_to_plot: Список имен оптимизаторов, для которых нужно показать графики.

    Возвращает:
        Список словарей, где каждый словарь содержит результаты одного эксперимента.
    """
    results_all = []

    for func_config in tqdm(experiment_configs, desc="Functions"):
        current_func = func_config["func"]
        current_grad = func_config["grad"]
        current_func_params = func_config["base_params"]

        unwrapped_func_for_plotting = current_func
        if hasattr(current_func, "_original_func"):
            unwrapped_func_for_plotting = getattr(current_func, "_original_func")

        current_plot_title_base = func_config["plot_title_base"]
        current_plot_lims = func_config["plot_lims"]

        for initial_point in tqdm(
            func_config["initial_points"],
            desc=f"Initial Points for {current_plot_title_base}",
            leave=False,
        ):
            for opt_setting in tqdm(
                optimizer_settings,
                desc=f"Optimizers (IP: {initial_point})",
                leave=False,
            ):
                final_opt_kwargs = opt_setting["kwargs"].copy()
                opt_name_adjusted = opt_setting["name"]

                res = run_experiment(
                    opt_name_adjusted,
                    opt_setting["func"],
                    current_func,
                    current_grad,
                    initial_point,
                    current_func_params,
                    func_config["clean_func_name"],
                    current_plot_title_base,  # Это будет plot_title_for_graphs
                    **final_opt_kwargs,
                )
                results_all.append(res)

                if res["history"]:
                    should_show_plot = (
                        show_plots_interactively
                        and opt_name_adjusted in optimizers_to_plot
                    )

                    # file path
                    func_dir_name = sanitize_for_path(res["plot_title_for_graphs"])
                    opt_dir_name = sanitize_for_path(opt_name_adjusted)

                    plot_save_dir = os.path.join(plots_dir, func_dir_name, opt_dir_name)
                    os.makedirs(plot_save_dir, exist_ok=True)

                    # generate filename
                    ip_str = (
                        np.array2string(initial_point, precision=1, separator="_")
                        .replace("[", "")
                        .replace("]", "")
                        .replace(" ", "")
                        .replace(".", "p")
                    )
                    ip_filename_part = sanitize_for_path(f"IP_{ip_str}")

                    plot_title_actual_graphs = f"{res['optimizer']} - {res['plot_title_for_graphs']} IP={res['initial_point']}"

                    plotting_func = unwrapped_func_for_plotting
                    plotting_params = current_func_params.copy()

                    # не забыть удалить  noise_sigma для передачи в оригинальную функцию
                    if "noise_sigma" in plotting_params:
                        plotting_params.pop("noise_sigma")

                    try:
                        # 2d график
                        plot_contour_and_path(
                            plotting_func,
                            plotting_params,
                            res["history"],
                            plot_title_actual_graphs,
                            current_plot_lims[0],
                            current_plot_lims[1],
                            save_path=os.path.join(
                                plot_save_dir, f"{ip_filename_part}_contour.png"
                            ),
                            show_plot=should_show_plot,
                        )
                    except Exception as e:
                        print(
                            f"Ошибка при построении контурного графика для {plot_title_actual_graphs}: {e}"
                        )

                    # --- График сходимости ---
                    try:
                        plot_convergence(
                            res["func_values_history"],
                            res["grad_norms_history"],
                            f"Сходимость: {plot_title_actual_graphs}",
                            save_path=os.path.join(
                                plot_save_dir, f"{ip_filename_part}_convergence.png"
                            ),
                            show_plot=should_show_plot,
                        )
                    except Exception as e:
                        print(
                            f"Ошибка при построении графика сходимости для {plot_title_actual_graphs}: {e}"
                        )

                    # --- 3D график  ---
                    try:
                        plot_surface_and_path_3d(
                            plotting_func,
                            plotting_params,
                            res["history"],
                            f"3D: {plot_title_actual_graphs}",
                            current_plot_lims[0],
                            current_plot_lims[1],
                            func_values_for_zlim=res["func_values_history"],
                            save_path=os.path.join(
                                plot_save_dir, f"{ip_filename_part}_surface3d.png"
                            ),
                            show_plot=should_show_plot,
                        )
                    except Exception as e:
                        print(
                            f"Ошибка при построении 3D графика для {plot_title_actual_graphs}: {e}"
                        )

    return results_all


def process_and_save_results(results_all: List[Dict[str, Any]], plots_dir: str):
    """
    Обрабатывает сводные результаты экспериментов, выводит их в консоль и сохраняет в CSV.

    Аргументы:
        results_all: Список словарей с результатами каждого эксперимента.
        plots_dir: Директория для сохранения итогового CSV-файла.
    """
    print("\n\n--- Сводная таблица результатов ---")
    if not results_all:
        print("Нет результатов для отображения.")
        return

    df_results = pd.DataFrame(results_all)

    columns_ordered = [
        "function_name",
        "func_params",
        "initial_point",
        "optimizer",
        "optimizer_params",
        "tolerance",
        "max_iterations",
        "final_point",
        "final_value",
        "iterations_taken",
        "func_evals",
        "grad_evals",
        "grad_norm",
        "status",
        "message",
    ]

    # инициализируем все столбцы
    for col in columns_ordered:
        if col not in df_results.columns:
            df_results[col] = pd.NA

    df_display = df_results[columns_ordered].copy()

    # читаемость для точек
    for col in ["initial_point", "final_point"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: np.array2string(np.array(x), precision=3, separator=", ")
                if isinstance(x, (list, np.ndarray))
                else x
            )

    if "final_value" in df_display.columns:
        df_display["final_value"] = df_display["final_value"].apply(
            lambda x: f"{x:.5f}"
            if isinstance(x, (float, np.floating)) and pd.notna(x)
            else x
        )
    if "grad_norm" in df_display.columns:
        df_display["grad_norm"] = df_display["grad_norm"].apply(
            lambda x: f"{x:.4e}"
            if isinstance(x, (float, np.floating)) and pd.notna(x)
            else x
        )

    if "func_params" in df_display.columns:
        df_display["func_params"] = df_display["func_params"].astype(str)
    if "optimizer_params" in df_display.columns:
        df_display["optimizer_params"] = df_display["optimizer_params"].astype(str)

    # отобразил pandas
    print(df_display)

    # сохраняем в csv
    try:
        csv_path = os.path.join(plots_dir, "optimization_results.csv")
        df_display.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"\nРезультаты также сохранены в CSV: {csv_path}")
    except Exception as e:
        print(f"\nНе удалось сохранить результаты в CSV: {e}")


"""


В целом:
*   `status`: Это числовой код, который указывает на результат завершения работы оптимизатора.
*   `message`: Это текстовое сообщение, которое даёт более детальное, человеко-читаемое описание этого результата.


Вот расшифровка для 

### `status = 0` (Успешное завершение или нормальное прекращение)
Это наиболее частый статус, который в основном указывает на то, что оптимизация завершилась штатно.
*   `Custom optimizer converged (assumed)`: Это сообщение от вашего самописного оптимизатора. Оно означает, что оптимизатор отработал все итерации без численных сбоев. Сходимость "предполагается", так как, вероятно, нет явной проверки на сходимость.
*   `Optimization terminated successfully.`: Стандартное сообщение от оптимизаторов `SciPy`, указывающее на успешное нахождение минимума.
*   `Gradient norm below tolerance.`: Сообщение от обертки для `Torch` оптимизаторов. Означает, что оптимизация остановлена, так как норма градиента стала меньше заданного порога, что является условием сходимости.
*   `Max iterations reached.`: Также от `Torch`. Оптимизатор остановился, потому что достиг максимального числа итераций. Это нормальное завершение, но не всегда означает, что найден оптимальный минимум.

### `status = 2` (Предупреждение)
Этот статус от `SciPy` указывает на то, что оптимизация завершилась, но результат может быть неидеальным.
*   `Maximum number of iterations has been exceeded.`: Оптимизатор не успел сойтись за отведенное количество итераций.
*   `Desired error not necessarily achieved due to precision loss.`: Требуемая точность не была достигнута, возможно, из-за проблем с численной устойчивостью или точностью вычислений.

### `status = -1` (Ошибка)
Этот статус указывает на провал оптимизации.
*   `Numerical issues (NaN/Inf in point, value, or grad_norm).`: Во время вычислений возникли численные проблемы, которые привели к появлению `NaN` (не число) или `inf` (бесконечность). Это прервало процесс оптимизации.


"""
