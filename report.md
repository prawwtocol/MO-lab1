## 1. Постановка задачи

Основная задача этой лабораторной работы - разработка методов градиентного спуска, исследование работоспособности различных методов градиентного спуска, реализованных руками и библиотечных. В частности, нас интересует градиентный спуск на основе одномерного поиска (золотое сечение, критерий Армихо), то есть методы первого и нулевого порядка. Исследование проводится как на очевидных функционных, так и на более сложных, мультимодальных и зашумленных.


## 2. Описание используемых методов

Были реализованы и исследованы следующие методы оптимизации.

### 2.1. Сделано ручками

#### 2.1.1. Градиентный спуск (`gradient_descent`)

Классический метод градиентного спуска первого порядка. Алгоритм итеративно обновляет текущую точку в направлении антиградиента:

$$ x_{k+1} = x_k - \alpha_k \nabla f(x_k) $$
где $ \alpha_k $ — длина шага (learning rate) на $ k $-й итерации.

Особенности:
- **Выбор шага:** Параметр `learning_rate` может быть как константой, так и функцией от номера итерации (learning rate scheduling). Были реализованы следующие стратегии:
    - **Константный шаг (Constant LR):**
      $$ \alpha_k = \alpha_0 $$
      где $ \alpha_0 $ — фиксированное значение на всех итерациях.
    - **Экспоненциальное затухание (Exponential Decay):**
      $$ \alpha_k = \alpha_0 \cdot \gamma^k $$
      где $ \gamma \in (0, 1) $ — фактор затухания. Шаг уменьшается на каждой итерации.
    - **Пошаговое уменьшение (Step Decay):**
      $$ \alpha_k = \alpha_0 \cdot \gamma^{\lfloor k / s \rfloor} $$
      где $ s $ — количество шагов, после которых происходит уменьшение `learning rate`. Шаг остается постоянным на протяжении $ s $ итераций, а затем умножается на $ \gamma $.
- **Критерии остановки:** Алгоритм останавливается при достижении максимального числа итераций, либо если норма градиента становится меньше заданного порога (`tolerance`).
- **Защита от расходимости:** Встроена проверка на появление `NaN` или `Infinity` в векторе градиента или координат точки, что предотвращает аварийное завершение при численной нестабильности.

#### 2.1.2. Градиентный спуск с поиском шага методом золотого сечения (`gradient_descent_with_line_search`)

Модификация градиентного спуска, в которой на каждой итерации решается задача одномерной минимизации для определения оптимальной длины шага $ \alpha_k $.
$$ \alpha_k = \arg\min_{\alpha > 0} f(x_k - \alpha \nabla f(x_k)) $$
Для решения этой задачи был реализован **метод золотого сечения** (`golden_section_search`) — алгоритм нулевого порядка, который не требует вычисления производных.

Преимущества:
- **Адаптивность:** Размер шага оптимально подбирается на каждой итерации, что особенно полезно на функциях со сложным рельефом.
- **Надежность:** Гарантируется уменьшение значения функции на каждом шаге (при условии точного одномерного поиска).

Недостаток:
- **Высокая стоимость итерации:** Поиск оптимального шага требует многократных вычислений целевой функции, что может быть затратно.

#### 2.1.3. Градиентный спуск с критерием Армихо (`gradient_descent_with_armijo_ls`)

Это еще один вариант градиентного спуска с адаптивным шагом. Вместо поиска точного минимума вдоль направления спуска, используется стратегия возврата (backtracking) для нахождения шага, удовлетворяющего **условию Армихо (достаточного убывания)**.
$$ f(x_k - \alpha \nabla f(x_k)) \le f(x_k) - c_1 \alpha \| \nabla f(x_k) \|^2 $$
Алгоритм начинает с некоторого начального шага `initial_alpha` и уменьшает его с коэффициентом `rho_armijo`, пока условие не будет выполнено.

Преимущества:
- **Баланс:** Обеспечивает достаточное убывание функции, но избегает дорогостоящего точного поиска, как в методе золотого сечения.
- **Эффективность:** Часто находит хороший шаг быстрее, чем точные методы линейного поиска.

Недостатки:
- **Чувствительность к гиперпараметрам:** Эффективность сильно зависит от выбора начального шага (`initial_alpha`), коэффициента `c1` и фактора уменьшения шага (`rho_armijo`). Неудачный подбор может привести к множеству лишних вычислений функции или к выбору неэффективного шага.
- **Неоптимальность шага:** Метод гарантирует лишь *достаточное*, а не оптимальное убывание. Это может привести к большему числу итераций по сравнению с методами, выполняющими более точный поиск.
- **Риск слишком малых шагов:** Алгоритм может находить и принимать шаги, которые удовлетворяют условию, но являются чрезмерно маленькими, что замедляет общую скорость сходимости.

### 2.2. Библиотечные методы

Для сравнения производительности использовались стандартные оптимизаторы из библиотеки `scipy.optimize`
- **Nelder-Mead:** Метод нулевого порядка (не использует градиент).
- **CG (Conjugate Gradient):** Метод сопряженных градиентов, эффективен для больших квадратичных задач.
- **BFGS (Broyden–Fletcher–Goldfarb–Shanno):** Один из самых популярных квазиньютоновских методов. Строит аппроксимацию обратной матрицы Гессе.

Методы scipy.optimize используются в проекте для вычислений и сравнения в таблицах (количество итераций, вызовы функций и т.д.), но их траектории не могут быть построены на графиках, потому что scipy не возвращает историю шагов, необходимую для визуализации пути оптимизации. На графиках отображаются только те методы, для которых вы вручную сохраняете историю в optimizers.py и lrs.py.


Также для демонстрации (с их помощью можно построить графики траектории ) использовались методы из библиотеки `torch`:
- **SGD (Stochastic Gradient Descent):** Реализация градиентного спуска в PyTorch.
- **Планировщики шага:** `ExponentialLR`, `StepLR` для демонстрации стандартных стратегий изменения learning rate.

## 3. Объекты исследования (Тестовые функции)

Для исследования поведения методов были выбраны функции с различными свойствами.

### 3.1. Эллиптический параболоид

$$ f(x, y) = c_1 x^2 + c_2 y^2 $$
Простая квадратичная функция с минимумом в точке (0, 0). Ключевым параметром является **число обусловленности** Гессиана, равное $ \max(c_1, c_2) / \min(c_1, c_2) $. Этот параметр характеризует "вытянутость" линий уровня функции. Исследование проводилось при числах обусловленности 10, 100 и 250.

### 3.2. Вытянутая квадратичная долина

Сложная квадратичная функция, имитирующая "овраг", что является классической проблемой для градиентного спуска. Имеет высокую степень обусловленности, что делает ее сложной для оптимизации.

### 3.3. Функция Бута

$$ f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2 $$
Простая, гладкая, унимодальная квадратичная функция с минимумом в точке (1, 3). Использовалась как базовый пример для демонстрации работы методов, а также в качестве основы для зашумленной функции.

### 3.4. Зашумленная функция Бута

Модификация функции Бута, к значению которой на каждом вызове добавлялся Гауссов шум:
$$ f_{noisy}(x, y) = f_{booth}(x, y) + \mathcal{N}(0, \sigma^2) $$
Этот объект исследования позволяет оценить робастность методов к неточностям в вычислении целевой функции.

### 3.5. Функция Экли

$$ f(x,y) = -20\exp(-0.2\sqrt{0.5(x^2+y^2)}) - \exp(0.5(\cos(2\pi x) + \cos(2\pi y))) + e + 20 $$
Классическая мультимодальная функция. Она имеет широкий, почти плоский "кратер" с множеством локальных минимумов и одним глобальным минимумом в точке (0, 0). Используется для проверки способности методов избегать "ловушек" локальных экстремумов.

## 4. Результаты исследования

Исследования проводились путем запуска каждого метода на каждой из тестовых функций с разных начальных точек. Результаты фиксировались в виде таблиц (количество итераций, вычислений функции и градиента) и графиков (линии уровня с траекториями движения методов).

### 4.1. Влияние числа обусловленности (Эллиптический параболоид)

- **Вывод:** Увеличение числа обусловленности критически сказывается на производительности простых методов градиентного спуска.
- **Наблюдения:**
    - При `cond=250` градиентный спуск с постоянным шагом и с простыми планировщиками (`ExponentialLR`, `StepLR`) расходился, что приводило к численным ошибкам (`NaN`/`Inf`).
    - В то же время, градиентный спуск с поиском шага методом золотого сечения (`LineSearch GS`) оставался стабильным и находил решение за малое число итераций. Это демонстрирует, что адаптация шага является мощным инструментом для борьбы с плохой обусловленностью.
    - Метод с критерием Армихо также сходился, но требовал значительно больше итераций, показывая меньшую эффективность в тяжелых условиях.
    - Методы `SciPy (BFGS, CG)` также показали высокую эффективность.

### 4.2. Овражные функции (Вытянутая долина)

- **Вывод:** Простой градиентный спуск неспособен эффективно оптимизировать "овражные" функции, в то время как адаптивные и квазиньютоновские методы справляются с этой задачей.
- **Наблюдения:**
    - `GD (Const LR)` быстро расходился, "отскакивая" от стенок оврага.
    - Методы с адаптивным шагом (`ExpDecay`, `StepDecay`, `LineSearch GS`, `Armijo LS`) успешно сходились. Способность подбирать разную длину шага позволила им двигаться вдоль дна оврага.
    - `SciPy (BFGS, CG)` показали наилучшие результаты, что ожидаемо, так как они строят информацию о кривизне второго порядка, эффективно определяя направление оврага.

### 4.3. Влияние шума в целевой функции (Зашумленная функция Бута)

- **Вывод:** Шум сильнее всего влияет на методы, которые интенсивно используют значения целевой функции для своей работы.
- **Наблюдения:**
    - У метода с поиском шага золотым сечением (`LineSearch GS`) количество вычислений функции `func_evals` выросло на порядки (с ~200 до ~76000). Алгоритм тратил огромные ресурсы, пытаясь найти "оптимальный" шаг на зашумленной поверхности.
    - Метод `Nelder-Mead` (нулевого порядка) не смог сойтись, так как полностью полагается на значения функции, которые его "обманывали".
    - `SciPy (BFGS, CG)` также испытывали трудности, выдавая предупреждения о потере точности.

### 4.4. Мультимодальные функции (Функция Экли)

- **Вывод:** Все исследованные методы являются локальными, однако адаптивный выбор шага может в некоторых случаях помочь "выпрыгнуть" из локального минимума.
- **Наблюдения:**
    - При старте из точки [2.5, 2.5] большинство методов попадали в ближайший локальный минимум.
    - Интересно, что `LineSearch GS` и `Armijo LS` из той же стартовой точки смогли найти глобальный минимум в (0, 0). Вероятно, большой шаг, выбранный на первых итерациях, позволил им "перепрыгнуть" барьер, отделяющий локальный минимум.
    - При старте вблизи глобального минимума, простой `GD` сходился очень медленно из-за малых градиентов в этой области.

## 5. Выводы


1.  **Простой градиентный спуск** с постоянным шагом является не очень надежным методом. Его производительность критически зависит от выбора шага и свойств целевой функции (особенно от числа обусловленности).
2.  **LR Schedulers**, такие как экспоненциальное или пошаговое уменьшение, повышают робастность градиентного спуска, но все еще требуют подбора гиперпараметров.
3.  **Градиентный спуск с линейным поиском** (на примере Золотого сечения и критерия Армихо) показал себя как мощный метод. Способность адаптивно подбирать шаг на каждой итерации позволяет ему эффективно решать задачи с плохой обусловленностью и даже иногда избегать локальных минимумов. При этом:
    - **Точный поиск (Золотое сечение)** может быть очень дорогим по числу вычислений функции, особенно на зашумленных данных.
    - **Неточный поиск (Армихо)** представляет собой хороший компромисс между эффективностью и затратами.
4.  **Библиотечные квазиньютоновские методы (`BFGS`, `CG`)** в большинстве случаев являются наиболее эффективным выбором для гладких функций, так как они используют информацию о кривизне второго порядка для более "умного" выбора направления и длины шага.
