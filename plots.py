"""
Утилитарный файл с методами отображения графиков и путей оптимизации
"""

import os
from typing import Optional, Tuple, Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from custom_types import FuncType, HistoryType


def plot_contour_and_path(
    func: FuncType,
    params: Dict[str, Any],
    history: Optional[HistoryType],
    title: str,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
    levels: int = 50,
    save_path: Optional[str] = None,
    show_plot: bool = False,
):
    """Рисует контурный график функции и путь оптимизации на нем"""
    try:
        fig = plt.figure(figsize=(10, 8))

        x = np.linspace(x_lim[0], x_lim[1], 200)
        y = np.linspace(y_lim[0], y_lim[1], 200)
        X, Y = np.meshgrid(x, y)

        Z_values = np.zeros_like(X)
        try:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    val = func(X[i, j], Y[i, j], **params)
                    Z_values[i, j] = val if np.isfinite(val) else np.nan
        except Exception as e:
            print(f"Ошибка при вычислении Z для контурного графика {title}: {e}")
            plt.close(fig)
            return

        if np.all(np.isnan(Z_values)):
            print(
                f"Все значения Z для контурного графика '{title}' являются NaN. График не будет построен."
            )
            plt.close(fig)
            return

        Z = Z_values

        min_z, max_z = np.nanmin(Z), np.nanmax(Z)
        levels_arg = levels
        if np.isfinite(min_z) and np.isfinite(max_z) and min_z < max_z:
            if min_z > 0 and max_z / min_z > 100:
                # Если большой разброс, используем логарифмическую шкалу
                levels_arg = np.logspace(
                    np.log10(min_z + 1e-9), np.log10(max_z), levels
                )
                # + 1e-9 чтобы избежать log(0)
            else:
                #  используем линейную шкалу
                levels_arg = np.linspace(min_z, max_z, levels)
        elif np.isfinite(min_z) and np.isfinite(max_z) and min_z == max_z:
            levels_arg = np.array([min_z])
        else:
            print(
                f"Не удалось определить уровни для контурного графика '{title}'. Используются уровни по умолчанию."
            )

        try:
            cp = plt.contour(X, Y, Z, levels=levels_arg, cmap="viridis")
            plt.clabel(cp, inline=1, fontsize=8)
            plt.colorbar(cp, label="Function Value")
        except Exception as e:
            print(f"Ошибка при построении контуров для '{title}': {e}")

        if history:
            try:
                history_np = np.array(history)
                # убираем NaN/Inf из истории
                valid_history_mask = np.all(np.isfinite(history_np), axis=1)
                history_np_filtered = history_np[valid_history_mask]

                if history_np_filtered.shape[0] > 0:
                    plt.plot(
                        history_np_filtered[:, 0],
                        history_np_filtered[:, 1],
                        "r-o",
                        markersize=3,
                        linewidth=1,
                        label="Path",
                    )
                    plt.plot(
                        history_np_filtered[0, 0],
                        history_np_filtered[0, 1],
                        "go",
                        markersize=8,
                        label="Start",
                    )
                    plt.plot(
                        history_np_filtered[-1, 0],
                        history_np_filtered[-1, 1],
                        "bo",
                        markersize=8,
                        label="End",
                    )
                else:
                    print(f"История для '{title}' пуста или содержит только NaN/Inf.")
            except Exception as e:
                print(f"Ошибка при построении пути для '{title}': {e}")

        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        try:
            plt.legend()
        except Exception as e:
            print(f"Ошибка при создании легенды для '{title}': {e}")
        plt.grid(True)

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                print(f"График сохранен: {save_path}")
            except Exception as e:
                print(f"Не удалось сохранить график {save_path}: {e}")

        if show_plot:
            plt.show()

        plt.close(fig)
    except Exception as e:
        print(f"Критическая ошибка при построении контурного графика '{title}': {e}")
        plt.close("all")


def plot_convergence(
    func_values_hist,
    grad_norms_hist,
    title,
    save_path: Optional[str] = None,
    show_plot: bool = False,
):
    """Рисует графики сходимости (значение функции и норма градиента)"""
    try:
        # убираем NaN и Inf
        valid_func_values = []
        if func_values_hist:
            valid_func_values = [v for v in func_values_hist if np.isfinite(v)]

        valid_grad_norms = []
        if grad_norms_hist:
            valid_grad_norms = [v for v in grad_norms_hist if np.isfinite(v)]

        if not valid_func_values and not valid_grad_norms:
            print(f"Нет валидных данных для построения графиков сходимости для {title}")
            return

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)

        if valid_func_values:
            axs[0].plot(valid_func_values, marker=".")
            axs[0].set_title("Значение функции от итерации")
            axs[0].set_xlabel("Итерация")
            axs[0].set_ylabel("f(x)")
            axs[0].grid(True)
        else:
            axs[0].text(
                0.5,
                0.5,
                "Нет данных по значениям функции",
                ha="center",
                va="center",
                transform=axs[0].transAxes,
            )

        if valid_grad_norms:
            axs[1].plot(valid_grad_norms, marker=".")
            axs[1].set_title("Норма градиента от итерации")
            axs[1].set_xlabel("Итерация")
            axs[1].set_ylabel("||∇f(x)||")
            if any(v > 0 for v in valid_grad_norms):
                try:
                    # пытаемся рисовать на логарифмической шкале
                    axs[1].set_yscale("log")
                except ValueError:
                    axs[1].set_yscale("linear")
                    print(
                        f"Предупреждение: не удалось установить логарифмическую шкалу для нормы градиента в '{title}'. Используется линейная."
                    )
            axs[1].grid(True)
        else:
            axs[1].text(
                0.5,
                0.5,
                "Нет данных по нормам градиента",
                ha="center",
                va="center",
                transform=axs[1].transAxes,
            )

        try:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        except ValueError as e:
            print(
                f"Предупреждение: Не удалось применить tight_layout для '{title}': {e}. График может выглядеть неоптимально."
            )

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                print(f"График сохранен: {save_path}")
            except Exception as e:
                print(f"Не удалось сохранить график {save_path}: {e}")

        if show_plot:
            plt.show()

        plt.close(fig)
    except Exception as e:
        print(f"Критическая ошибка при построении графика сходимости '{title}': {e}")
        plt.close("all")


def plot_surface_and_path_3d(
    func,
    params,
    history,
    title,
    x_lim,
    y_lim,
    z_lim=None,
    func_values_for_zlim=None,
    save_path: Optional[str] = None,
    show_plot: bool = False,
):
    """Рисует 3D поверхность функции и путь оптимизации"""
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        x_surf = np.linspace(x_lim[0], x_lim[1], 70)
        y_surf = np.linspace(y_lim[0], y_lim[1], 70)
        X_surf, Y_surf = np.meshgrid(x_surf, y_surf)

        Z_surf = np.full_like(X_surf, np.nan)
        try:
            for i in range(X_surf.shape[0]):
                for j in range(X_surf.shape[1]):
                    val = func(X_surf[i, j], Y_surf[i, j], **params)
                    if np.isfinite(val):
                        Z_surf[i, j] = val
        except Exception as e:
            print(f"Ошибка при вычислении Z для 3D поверхности {title}: {e}")

        # Строим поверхность
        if not np.all(np.isnan(Z_surf)):
            try:
                ax.plot_surface(
                    X_surf,
                    Y_surf,
                    Z_surf,
                    cmap="viridis",
                    edgecolor="none",
                    alpha=0.6,
                    rstride=3,
                    cstride=3,
                )
            except Exception as e:
                print(f"Ошибка при построении 3D поверхности для '{title}': {e}")
        else:
            print(
                f"Все значения Z для 3D поверхности '{title}' являются NaN. Поверхность не будет построена."
            )

        if history:
            history_np = np.array(history)
            # убираем NaN/Inf из истории
            valid_history_mask = np.all(np.isfinite(history_np), axis=1)
            history_np_filtered = history_np[valid_history_mask]

            if history_np_filtered.shape[0] > 0:
                x_path = history_np_filtered[:, 0]
                y_path = history_np_filtered[:, 1]

                z_path = np.full_like(x_path, np.nan)
                try:
                    for k in range(len(x_path)):
                        val = func(x_path[k], y_path[k], **params)
                        if np.isfinite(val):
                            z_path[k] = val
                except Exception as e:
                    print(f"Ошибка при вычислении Z для пути в 3D графике {title}: {e}")

                # убираем точки пути, у которых NaN
                valid_path_mask = np.isfinite(z_path)
                if np.any(valid_path_mask):
                    ax.plot(
                        x_path[valid_path_mask],
                        y_path[valid_path_mask],
                        z_path[valid_path_mask],
                        "r-o",
                        markersize=3,
                        linewidth=1.5,
                        label="Path",
                    )

                    # Начальная и конечная точки другим цветом
                    if np.isfinite(z_path[valid_path_mask][0]):
                        ax.plot(
                            [x_path[valid_path_mask][0]],
                            [y_path[valid_path_mask][0]],
                            [z_path[valid_path_mask][0]],
                            "go",
                            markersize=8,
                            label="Start",
                        )
                    if np.isfinite(z_path[valid_path_mask][-1]):
                        ax.plot(
                            [x_path[valid_path_mask][-1]],
                            [y_path[valid_path_mask][-1]],
                            [z_path[valid_path_mask][-1]],
                            "bo",
                            markersize=8,
                            label="End",
                        )
                else:
                    print(f"Не удалось рассчитать Z для точек пути в 3D для '{title}'.")

            else:
                print(
                    f"История для 3D графика '{title}' пуста или содержит только NaN/Inf."
                )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("x1", fontsize=12)
        ax.set_ylabel("x2", fontsize=12)
        ax.set_zlabel("f(x1, x2)", fontsize=12)

        if z_lim:
            try:
                if np.all(np.isfinite(z_lim)):
                    ax.set_zlim(z_lim)
            except Exception as e:
                print(f"Ошибка при установке z_lim для '{title}': {e}")
        elif func_values_for_zlim is not None:
            valid_f_values = [v for v in func_values_for_zlim if np.isfinite(v)]
            if valid_f_values:
                try:
                    min_z_hist = np.min(valid_f_values)
                    max_z_hist = np.max(valid_f_values)
                    padding = (
                        (max_z_hist - min_z_hist) * 0.1
                        if max_z_hist > min_z_hist
                        else 1.0
                    )
                    final_min_z = min_z_hist - padding if min_z_hist > 0 else -padding
                    final_max_z = max_z_hist + padding
                    if (
                        np.isfinite(final_min_z)
                        and np.isfinite(final_max_z)
                        and final_min_z < final_max_z
                    ):
                        ax.set_zlim(final_min_z, final_max_z)
                except Exception as e:
                    print(
                        f"Ошибка при установке автоматических пределов по Z для '{title}': {e}"
                    )
            else:
                print(
                    f"Предупреждение: не удалось установить автоматические пределы по Z для '{title}'."
                )

        try:
            ax.legend()
            ax.view_init(elev=25, azim=-120)
        except Exception as e:
            print(f"Ошибка при настройке легенды или вида для '{title}': {e}")

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                print(f"График сохранен: {save_path}")
            except Exception as e:
                print(f"Не удалось сохранить график {save_path}: {e}")

        if show_plot:
            plt.show()

        plt.close(fig)
    except Exception as e:
        print(f"Критическая ошибка при построении 3D графика '{title}': {e}")
        plt.close("all")
