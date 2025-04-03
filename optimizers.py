from typing import Callable, Dict, Any, Tuple, Union

import numpy as np

from custom_types import (
    FuncType,
    GradType,
    LearningRateScheduler,
    HistoryType,
    ValuesType,
    PointType,
)
from util import get_expected_params


def gradient_descent(
    func: FuncType,
    grad_func: GradType,
    initial_point: PointType,
    params: Dict[str, Any],
    learning_rate: Union[float, LearningRateScheduler],
    n_iterations: int,
    tolerance: float,
) -> Tuple[PointType, HistoryType, ValuesType, ValuesType, int, int]:
    """
    Реализация градиентного спуска. Это метод первого порядка, То есть используясь значение функции в текущей точке и градиент в текущей точке для определения направления следующего шага.

    Аргументы:
    func -- оптимизируемая функция (принимает координаты и params) (f)
    grad_func -- функция, вычисляющая градиент (принимает координаты и params) (∇f)
    ∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
    где ∂f/∂xᵢ - частная производная f по переменной xᵢ.

    initial_point -- начальная точка (np.array) (x_0)
    params -- словарь дополнительных параметров для func и grad_func
    learning_rate -- скорость обучения (float или callable для LRS) (α)
    n_iterations -- максимальное количество итераций (int) -- Ограничивает максимальное количество итераций во всех основных функциях, предотвращая бесконечные циклы.
    tolerance -- допуск для критерия остановки по норме градиента (float) -- Останавливает алгоритм, если норма градиента становится меньше этого значения, что указывает на достижение минимума.

    Возвращает:
    point -- найденная точка минимума (np.array)
    history -- список точек, пройденных алгоритмом (list of np.array)
    func_values -- список значений функции на каждой итерации (list of floats)
    grad_norms -- список норм градиента на каждой итерации (list of floats)
    n_func_evals -- количество вычислений целевой функции
    n_grad_evals -- количество вычислений градиента

    Критерии остановки:
    * n_iterations
    * tolearance
    * Проверка на расходимость: gradient_descent прерывается, если градиент или координаты точки содержат NaT или inf, защищая от численной нестабильности.

    Преимущества: Обычно они сходятся быстрее, чем методы нулевого порядка, при условии, что градиент доступен и его вычисление не слишком затратно.

    """

    n_func_evals = 0
    n_grad_evals = 0

    point = np.array(
        initial_point, dtype=float
    )  # Начинаем с той начальной точки, которую нам задали
    history: HistoryType = [point.copy()]

    expected_func_params = get_expected_params(func, params)
    expected_grad_params = get_expected_params(grad_func, params)

    current_f_val = func(point[0], point[1], **expected_func_params)
    n_func_evals += 1
    func_values: ValuesType = [current_f_val]

    grad = grad_func(
        point[0], point[1], **expected_grad_params
    )  # Находим градиент ∇f(x_k) в текущей точке.
    n_grad_evals += 1
    grad_norms: ValuesType = [np.linalg.norm(grad)]

    # Превращаем lr в callable даже если он float для единообразия
    lr_scheduler: LearningRateScheduler
    if callable(learning_rate):
        lr_scheduler = learning_rate
    else:
        lr_scheduler = lambda iteration: float(learning_rate)

    for i in range(
        n_iterations
    ):  # Работаем пока не достигнем максимального числа итераций
        if (
            np.linalg.norm(grad) < tolerance
        ):  # Если норма градиента становится очень маленькой (близкой к нулю), то мы на "ровном месте" (в локальном минимуме или на плато) и останавливаемся
            break

        # Проверка на расходимость градиента
        if not np.all(np.isfinite(grad)):
            print(
                f"  Внимание: Градиент содержит NaN или Inf на итерации {i}. Остановка."
            )
            break

        lr = lr_scheduler(i)  # Выбираем размер шага
        point = point - lr * grad  # Делаем шаг: x_{k+1} = x_k - α * ∇f(x_k).

        # Проверка на расходимость в точке
        if not np.all(np.isfinite(point)):
            print(
                f"  Внимание: Точка содержит NaN или Inf на итерации {i} после шага. Остановка."
            )
            history.append(point.copy())
            func_values.append(np.nan)
            grad_norms.append(np.nan)
            break

        history.append(point.copy())

        current_f_val = func(point[0], point[1], **expected_func_params)
        n_func_evals += 1
        func_values.append(current_f_val)

        grad = grad_func(
            point[0], point[1], **expected_grad_params
        )  # Находим следующий градиент ∇f(x_k)
        n_grad_evals += 1
        grad_norms.append(np.linalg.norm(grad))

    return point, history, func_values, grad_norms, n_func_evals, n_grad_evals


def golden_section_search(
    func_1d: Callable[[float, Any], float],
    a: float,
    b: float,
    tol: float = 1e-5,
    max_iter: int = 100,
    args: Tuple = (),
) -> Tuple[float, float, int, int]:
    """
    Метод золотого сечения для одномерной минимизации. Это не метод градиентного спуска, а метод нулевого порядка, который будет нашим helper'ом для поиска оптимального шага.
    Это процедура нахождения оптимального размера шага α вдоль выбранного направления (обычно антиградиента).
    Находит такое значение α > 0, которое минимизирует f(x + α·d) по α.

    Аргументы:
    func_1d -- одномерная функция для минимизации (принимает скаляр x и args)
    a, b -- границы интервала поиска (a < b)
    tol -- допуск для остановки (разница между границами интервала)
    max_iter -- максимальное количество итераций
    args -- кортеж дополнительных аргументов для func_1d

    Возвращает:
    x_min -- точка минимума
    f_min -- значение функции в точке минимума
    iterations -- количество выполненных итераций
    n_func_evals_gs -- количество вычислений функции func_1d внутри GS

    Преимущества:
    Адаптивность: Размер шага выбирается оптимально для каждой итерации.
    Надежность: Можно гарантировать уменьшение функции на каждой итерации.
    Эффективность: Метод может сходиться за меньшее число итераций градиентного спуска.

    Работает даже когда:
    Производную невозможно или очень сложно вычислить.
    Функция является "черным ящиком" (мы можем только вычислять её значение для данных входных параметров).
    Функция зашумлена или не является гладкой.

    """

    inv_phi = (np.sqrt(5) - 1) / 2
    inv_phi_sq = (3 - np.sqrt(5)) / 2

    n_func_evals_gs = 0

    x1 = b - inv_phi * (b - a)
    x2 = a + inv_phi * (b - a)

    f1 = func_1d(x1, *args)
    n_func_evals_gs += 1
    f2 = func_1d(x2, *args)
    n_func_evals_gs += 1

    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        if abs(b - a) < tol:
            break

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - inv_phi * (b - a)
            f1 = func_1d(x1, *args)
            n_func_evals_gs += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + inv_phi * (b - a)
            f2 = func_1d(x2, *args)
            n_func_evals_gs += 1

    x_min = (a + b) / 2
    f_min_at_center = func_1d(x_min, *args)
    n_func_evals_gs += 1

    return x_min, f_min_at_center, iterations, n_func_evals_gs


"""
Методы линейных поисков:
    Преимущества: Могут обеспечить хорошую сходимость, так как на каждом шаге выбирается "наилучший" возможный шаг в данном направлении.
    Недостатки: Вычисление оптимального шага на каждой итерации может быть затратным (требует многократных вычислений функции).
"""


def gradient_descent_with_line_search(
    func: FuncType,
    grad_func: GradType,
    initial_point: PointType,
    params: Dict[str, Any],
    n_iterations: int,
    tolerance: float,
    line_search_tol: float,
) -> Tuple[PointType, HistoryType, ValuesType, ValuesType, int, int]:
    """
    Градиентный спуск с использованием метода золотого сечения для поиска оптимального шага. Вместо того чтобы заранее определять шаг, на каждой итерации мы задаемся вопросом: "Вдоль направления спуска -∇f(x) какой длины шаг будет самым лучшим?".
    """
    n_total_func_evals = 0
    n_total_grad_evals = 0

    # Аналогично предыдущей реализации
    point = np.array(initial_point, dtype=float)
    history: HistoryType = [point.copy()]

    # Фильтруем параметры для func и grad_func
    expected_func_params = get_expected_params(func, params)
    expected_grad_params = get_expected_params(grad_func, params)

    current_f_val = func(point[0], point[1], **expected_func_params)
    n_total_func_evals += 1
    func_values: ValuesType = [current_f_val]

    grad = grad_func(point[0], point[1], **expected_grad_params)
    n_total_grad_evals += 1
    grad_norms: ValuesType = [np.linalg.norm(grad)]

    for i in range(n_iterations):
        if np.linalg.norm(grad) < tolerance:
            break

        # Проверка на расходимость градиента
        if not np.all(np.isfinite(grad)):
            print(
                f"  Внимание: Градиент содержит NaN или Inf на итерации {i}. Остановка."
            )
            break

        direction = -grad

        # Для этой функции мы будем искать оптимальное alpha; она выдает высоту горы, если сделать шаг длины alpha
        def func_1d(alpha: float, p: PointType, d: PointType) -> float:
            new_point = p + alpha * d
            return func(new_point[0], new_point[1], **expected_func_params)

        optimal_alpha, _, _, n_fevals_gs = golden_section_search(
            func_1d, a=0, b=1.0, tol=line_search_tol, args=(point, direction)
        )
        n_total_func_evals += n_fevals_gs

        point = point + optimal_alpha * direction

        # Проверка на расходимость в точке
        if not np.all(np.isfinite(point)):
            print(
                f"  Внимание: Точка содержит NaN или Inf на итерации {i} после шага. Остановка."
            )
            history.append(point.copy())
            func_values.append(np.nan)
            grad_norms.append(np.nan)
            break

        history.append(point.copy())

        # Пересчитываем значение функции и градиент в новой точке
        current_f_val = func(point[0], point[1], **expected_func_params)
        n_total_func_evals += 1
        func_values.append(current_f_val)

        grad = grad_func(point[0], point[1], **expected_grad_params)
        n_total_grad_evals += 1
        grad_norms.append(np.linalg.norm(grad))

    return (
        point,
        history,
        func_values,
        grad_norms,
        n_total_func_evals,
        n_total_grad_evals,
    )


def gradient_descent_with_armijo_ls(
    func: FuncType,
    grad_func: GradType,
    initial_point: PointType,
    params: Dict[str, Any],
    n_iterations: int,
    tolerance: float,
    initial_alpha: float = 1.0,
    c1_armijo: float = 1e-4,
    rho_armijo: float = 0.5,
    alpha_min_armijo: float = 1e-10,
) -> Tuple[PointType, HistoryType, ValuesType, ValuesType, int, int]:
    """
    Градиентный спуск с линейным поиском по правилу Армихо (backtracking).

    Дополнительные аргументы:
    initial_alpha -- начальный размер шага для поиска
    c1_armijo -- константа для условия Армихо (обычно малое положительное число)
    rho_armijo -- коэффициент уменьшения шага (0 < rho < 1)
    alpha_min_armijo -- минимальный порог для шага, чтобы избежать бесконечного цикла
    """
    n_total_func_evals = 0
    n_total_grad_evals = 0

    # Аналогично предыдущей реализации
    point = np.array(initial_point, dtype=float)
    history: HistoryType = [point.copy()]

    expected_func_params = get_expected_params(func, params)
    expected_grad_params = get_expected_params(grad_func, params)

    current_f_val = func(point[0], point[1], **expected_func_params)
    n_total_func_evals += 1
    func_values: ValuesType = [current_f_val]

    grad = grad_func(point[0], point[1], **expected_grad_params)
    n_total_grad_evals += 1
    grad_norms: ValuesType = [np.linalg.norm(grad)]

    for i in range(n_iterations):
        if np.linalg.norm(grad) < tolerance:
            break

        # Проверка на расходимость градиента
        if not np.all(np.isfinite(grad)):
            print(
                f"  Внимание: Градиент содержит NaN или Inf на итерации {i}. Остановка."
            )
            break

        # Теперь ищем оптимальный шаг по правилу Армихо
        alpha = initial_alpha
        direction = -grad

        f_k = current_f_val
        # dot(grad, pk) = (∇f(x_k)^T p_k) -- производная по направлению
        grad_dot_pk = np.dot(grad, direction)

        while alpha > alpha_min_armijo:
            # f_new = f(x_k + alpha*p_k)
            f_new = func(
                point[0] + alpha * direction[0],
                point[1] + alpha * direction[1],
                **expected_func_params,
            )
            n_total_func_evals += 1

            # условие Армихо.
            # f_new должна быть ниже  правой части неравенства.
            if f_new <= f_k + c1_armijo * alpha * grad_dot_pk:
                break  # Шаг найден, успех

            # Если условие не выполнено, шаг слишком большой, уменьшаем его.
            alpha *= rho_armijo
        else:
            alpha = 0  # Не удалось найти подходящий шаг, останавливаемся

        if alpha == 0:
            break

        # Обновляем точку с найденным шагом
        point = point + alpha * direction

        # Проверка на расходимость в точке
        if not np.all(np.isfinite(point)):
            print(
                f"  Внимание: Точка содержит NaN или Inf на итерации {i} после шага. Остановка."
            )
            history.append(point.copy())
            func_values.append(np.nan)
            grad_norms.append(np.nan)
            break

        history.append(point.copy())

        # Обновляем значения для следующей итерации
        current_f_val = func(point[0], point[1], **expected_func_params)
        n_total_func_evals += 1
        func_values.append(current_f_val)

        grad = grad_func(point[0], point[1], **expected_grad_params)
        n_total_grad_evals += 1
        grad_norms.append(np.linalg.norm(grad))

    return (
        point,
        history,
        func_values,
        grad_norms,
        n_total_func_evals,
        n_total_grad_evals,
    )
