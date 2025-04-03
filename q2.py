import os

import numpy as np

from config import PLOTS_DIR
from functions import (
    booth_function,
    booth_function_gradient,
    elongated_valley_quadratic,
    elongated_valley_quadratic_gradient,
    ackley_function,
    ackley_function_gradient,
    noisy_function_wrapper,
    elliptic_paraboloid,
    elliptic_paraboloid_gradient
)
from lrs import constant_lr, exponential_decay_lr, step_decay_lr
from main import process_and_save_results
from main import run_all_experiments_and_plot
from optimizers import (
    gradient_descent,
    gradient_descent_with_line_search,
    gradient_descent_with_armijo_ls,
)

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
    print(f"Директория {PLOTS_DIR} создана для сохранения графиков.")

OPTIMIZERS_TO_PLOT = [
    "GD (Armijo LS)",
    "SciPy (BFGS)",
    "SciPy (Nelder-Mead)",
    "SciPy (CG)",
    "Torch (SGD Const LR)",
    "GD (Const LR=0.001)",
    "GD (Const LR=0.01)",
    "GD (ExpDecay LR)",
    "GD (StepDecay LR)",
    "GD (LineSearch GS)",
]

SHOW_PLOTS_INTERACTIVELY = False
print(f"\n--- ВАЖНО: Графики будут построены для оптимизаторов: {OPTIMIZERS_TO_PLOT} ---\n")
print(f"--- Интерактивный показ графиков: {'ВКЛЮЧЕН' if SHOW_PLOTS_INTERACTIVELY else 'ОТКЛЮЧЕН'} ---\n")

experiment_configs = []

booth_plot_lims = ((-5, 7), (-2, 8))
booth_initial_points = [
    np.array([-3.0, 7.0]),
    np.array([5.0, -1.0]),
    np.array([-4.0, -1.0])
]
booth_configs = {
    'func': booth_function,
    'grad': booth_function_gradient,
    'base_params': {},
    'clean_func_name': "Функция Бута",
    'plot_title_base': "Функция Бута",
    'plot_lims': booth_plot_lims,
    'initial_points': booth_initial_points
}
experiment_configs.append(booth_configs)

noise_sigmas_booth = [0.1, 0.5]
for sigma in noise_sigmas_booth:
    noisy_booth_params = {'noise_sigma': sigma}
    experiment_configs.append({
        'func': noisy_function_wrapper(booth_function, sigma),
        'grad': booth_function_gradient,  # Используем оригинальный градиент для зашумленной функции
        'base_params': noisy_booth_params,  # Для логирования и передачи wrapper'у
        'clean_func_name': f"Зашумленная Функция Бута (σ={sigma})",
        'plot_title_base': f"Зашумленная Функция Бута (σ={sigma})",
        'plot_lims': booth_plot_lims,
        'initial_points': booth_initial_points
    })

valley_plot_lims = ((-2, 12), (-2, 12))
valley_base_params = {'x_c': 5, 'y_c': 5, 'a1': 1, 'a2': 100, 'a3': 10}  # Высокое число обусловленности
valley_initial_points = [
    np.array([0.0, 0.0]),
    np.array([10.0, 2.0]),
    np.array([2.0, 10.0])
]
valley_configs = {
    'func': elongated_valley_quadratic,
    'grad': elongated_valley_quadratic_gradient,
    'base_params': valley_base_params,
    'clean_func_name': "Вытянутая квадратичная долина",
    'plot_title_base': f"Вытянутая квадр. долина (a1={valley_base_params['a1']},a2={valley_base_params['a2']},a3={valley_base_params['a3']})",
    'plot_lims': valley_plot_lims,
    'initial_points': valley_initial_points
}
experiment_configs.append(valley_configs)

paraboloid_plot_lims = ((-20, 20), (-20, 20))
paraboloid_params_sets = [
    {'c1': 1, 'c2': 10, 'cond': 10},  # Умеренно
    {'c1': 1, 'c2': 100, 'cond': 100},  # Плохо обусловленный
    {'c1': 1, 'c2': 250, 'cond': 250}  # Очень плохо обусловленный
]
for pp_set in paraboloid_params_sets:
    experiment_configs.append({
        'func': elliptic_paraboloid,
        'grad': elliptic_paraboloid_gradient,
        'base_params': {'c1': pp_set['c1'], 'c2': pp_set['c2'], 'cond': pp_set['cond']},
        'clean_func_name': "Эллиптический параболоид",
        'plot_title_base': f"Эллиптич. параболоид (cond={pp_set['cond']})",
        'plot_lims': paraboloid_plot_lims,
        'initial_points': [
            np.array([15.0, 15.0]),
            np.array([-10.0, 18.0]),
        ]
    })

# 4. Функция Экли (мультимодальная)
ackley_plot_lims = ((-5, 5), (-5, 5))
ackley_initial_points = [
    np.array([2.5, 2.5]), np.array([-3.0, 1.5]), np.array([0.1, -0.2]), np.array([-4, -4])
]
ackley_c_params = [np.pi, 2 * np.pi, 4 * np.pi]  # Разная "волнистость"

for c_val in ackley_c_params:
    ackley_base_params = {'a': 20, 'b': 0.2, 'c': c_val}
    experiment_configs.append({
        'func': ackley_function,
        'grad': ackley_function_gradient,
        'base_params': ackley_base_params,
        'clean_func_name': "Функция Экли",

        'plot_title_base': f"Функция Экли (a=20,b=0.2,c={c_val / np.pi:.1f}π)",
        'plot_lims': ackley_plot_lims,
        'initial_points': ackley_initial_points
    })

optimizer_settings = []

common_gd_params = {'n_iterations': 2000, 'tolerance': 1e-6}

exp_decay_initial_lr = 0.01
exp_decay_rate = 0.95
exp_decay_steps = 100
exp_decay_params_str = f"exponential_decay_lr({exp_decay_initial_lr}, {exp_decay_rate}, {exp_decay_steps})"

step_decay_initial_lr = 0.01
step_decay_drop_rate = 0.5
step_decay_epochs_drop = 500
step_decay_params_str = f"step_decay_lr({step_decay_initial_lr}, {step_decay_drop_rate}, {step_decay_epochs_drop})"

optimizer_settings.extend([
    {'name': "GD (Const LR=0.001)", 'func': gradient_descent,
     'kwargs': {'learning_rate': constant_lr(0.001), 'lr_details': "constant_lr(0.001)", **common_gd_params}},
    {'name': "GD (Const LR=0.01)", 'func': gradient_descent,
     'kwargs': {'learning_rate': constant_lr(0.01), 'lr_details': "constant_lr(0.01)", **common_gd_params}},
    {'name': "GD (ExpDecay LR)", 'func': gradient_descent,
     'kwargs': {'learning_rate': exponential_decay_lr(exp_decay_initial_lr, exp_decay_rate, exp_decay_steps),
                'lr_details': exp_decay_params_str, **common_gd_params}},
    {'name': "GD (StepDecay LR)", 'func': gradient_descent,
     'kwargs': {'learning_rate': step_decay_lr(step_decay_initial_lr, step_decay_drop_rate, step_decay_epochs_drop),
                'lr_details': step_decay_params_str, **common_gd_params}},
    {'name': "GD (LineSearch GS)", 'func': gradient_descent_with_line_search,
     'kwargs': {'line_search_tol': 1e-7, 'lr_details': "GoldenSectionLineSearch", **common_gd_params}},
    {'name': "GD (Armijo LS)", 'func': gradient_descent_with_armijo_ls,
     'kwargs': {'initial_alpha': 1.0, 'c1_armijo': 1e-4, 'rho_armijo': 0.5, 'alpha_min_armijo': 1e-10,
                'lr_details': "ArmijoLS(init_a=1.0,c1=1e-4,rho=0.5)", **common_gd_params}},
])

# SciPy методы
common_scipy_params = {'n_iterations': 1000, 'tolerance': 1e-6}
optimizer_settings.extend([
    {'name': "SciPy (Nelder-Mead)", 'func': None, 'kwargs': {'method': 'Nelder-Mead', **common_scipy_params}},
    {'name': "SciPy (BFGS)", 'func': None, 'kwargs': {'method': 'BFGS', **common_scipy_params}},
    {'name': "SciPy (CG)", 'func': None, 'kwargs': {'method': 'CG', **common_scipy_params}},
])

# PyTorch методы
common_torch_params = {'n_iterations': 20000, 'tolerance': 1e-6}
optimizer_settings.extend([
    {'name': "Torch (SGD Const LR)", 'func': None, 'kwargs': {
        **common_torch_params,
        'lr_details': {'type': 'constant', 'initial_lr': 0.001}
    }},
    {'name': "Torch (SGD ExpDecay LR)", 'func': None, 'kwargs': {
        **common_torch_params,
        'lr_details': {'type': 'exponential', 'initial_lr': exp_decay_initial_lr, 'decay_rate': exp_decay_rate,
                       'decay_steps': exp_decay_steps}
    }},
    {'name': "Torch (SGD StepDecay LR)", 'func': None, 'kwargs': {
        **common_torch_params,
        'lr_details': {'type': 'step', 'initial_lr': step_decay_initial_lr, 'drop_rate': step_decay_drop_rate,
                       'epochs_drop': step_decay_epochs_drop}
    }},
])

results_all = run_all_experiments_and_plot(experiment_configs, optimizer_settings, PLOTS_DIR, SHOW_PLOTS_INTERACTIVELY,
                                           OPTIMIZERS_TO_PLOT)

process_and_save_results(results_all, PLOTS_DIR)
