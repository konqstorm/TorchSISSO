import sys
import re
import torch
import numpy as np
from .import FeatureSpaceConstruction as fc
from .import DimensionalFeatureSpaceConstruction as dfc
from .Regressor import Regressor
from .Regressor_dimension import Regressor as Regressor_dim


class SissoModel:
    def __init__(self, data, operators=None, multi_task=None, n_expansion=None, n_term=None, k=20,
                 device='cpu', use_gpu=False, relational_units=None, initial_screening=None,
                 dimensionality=None, output_dim=None, regressor_screening=None,
                 custom_unary_functions=None, custom_binary_functions=None):

        if operators is None:
            sys.exit('!! Please provide the valid operator set!!')

        self.operators = operators
        self.df = data
        self.no_of_operators = n_expansion if n_expansion is not None else 3
        self.device = device
        self.dimension = n_term if n_term is not None else 3
        self.sis_features = k
        self.relational_units = relational_units
        self.initial_screening = initial_screening
        self.dimensionality = dimensionality
        self.output_dim = output_dim
        self.regressor_screening = regressor_screening
        self.use_gpu = use_gpu
        self.custom_unary_functions = custom_unary_functions
        self.custom_binary_functions = custom_binary_functions
        self.multi_task = multi_task

        if multi_task is not None:
            self.multi_task_target = multi_task[0]
            self.multi_task_features = multi_task[1]

    def fit(self):
        # Включаем GPU если нужно
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'

        if self.dimensionality is None:
            if self.multi_task is not None:
                print('Performing MultiTask Symbolic regression!!..')
                equations, rmses, r2s = [], [], []

                # ОПТИМИЗАЦИЯ: Если набор фичей для всех задач одинаков,
                # строим пространство признаков ОДИН РАЗ (основной боттлнек)
                # Здесь предполагаем, что если списки фичей одинаковы, мы экономим время.
                for i in range(len(self.multi_task_target)):
                    print(
                        f'Performing symbolic regression of target {i+1}....')

                    target_col = self.multi_task_target[i]
                    feature_cols = self.multi_task_features[i]

                    # Выбираем данные
                    cols_to_use = [target_col] + feature_cols
                    df_sub = self.df[cols_to_use]

                    # 1. Генерация признаков
                    constructor = fc.feature_space_construction(
                        self.operators, df_sub, self.no_of_operators,
                        self.device, self.initial_screening,
                        self.custom_unary_functions, self.custom_binary_functions
                    )
                    x, y, names = constructor.feature_space()

                    # 2. Регрессия
                    reg = Regressor(x, y, names, self.dimension,
                                    self.sis_features, self.device, self.use_gpu)
                    rmse, equation, r2, final_eq = reg.regressor_fit()

                    equations.append(final_eq)
                    rmses.append(rmse)
                    r2s.append(r2)

                    # Очистка памяти GPU после каждой задачи
                    if self.device == 'cuda':
                        del x, y, reg
                        torch.cuda.empty_cache()

                return rmses, equations, r2s

            else:
                # Обычный режим (одна задача)
                constructor = fc.feature_space_construction(
                    self.operators, self.df, self.no_of_operators,
                    self.device, self.initial_screening,
                    self.custom_unary_functions, self.custom_binary_functions
                )
                x, y, names = constructor.feature_space()

                reg = Regressor(x, y, names, self.dimension,
                                self.sis_features, self.device, self.use_gpu)
                return reg.regressor_fit()

        else:
            # Режим с учетом размерности (Dimensionality)
            if self.multi_task is not None:
                print('Performing MultiTask Dimensional Symbolic regression!!..')
                equations, rmses, r2s = [], [], []
                for i in range(len(self.multi_task_target)):
                    target_col = self.multi_task_target[i]
                    feature_cols = self.multi_task_features[i]
                    df_sub = self.df[[target_col] + feature_cols]

                    constructor = dfc.feature_space_construction(
                        df_sub, self.operators, self.relational_units,
                        self.initial_screening, self.no_of_operators,
                        self.device, self.dimensionality
                    )
                    x, y, names, dim = constructor.feature_expansion()

                    reg = Regressor_dim(x, y, names, dim, self.dimension, self.sis_features,
                                        self.device, self.output_dim, self.regressor_screening, self.use_gpu)
                    rmse, equation, r2 = reg.regressor_fit()

                    equations.append(equation)
                    rmses.append(rmse)
                    r2s.append(r2)
                return rmses, equations, r2s
            else:
                constructor = dfc.feature_space_construction(
                    self.df, self.operators, self.relational_units,
                    self.initial_screening, self.no_of_operators,
                    self.device, self.dimensionality
                )
                x, y, names, dim = constructor.feature_expansion()
                reg = Regressor_dim(x, y, names, dim, self.dimension, self.sis_features,
                                    self.device, self.output_dim, self.regressor_screening, self.use_gpu)
                return reg.regressor_fit()

    # Заранее компилируем правила замены для evaluate
    REPLACEMENTS = {
        r'\bexp\b': 'np.exp', r'\bcos\b': 'np.cos', r'\bsin\b': 'np.sin',
        r'\btan\b': 'np.tan', r'\bcsc\b': '1/np.sin', r'\bsec\b': '1/np.cos',
        r'\bcot\b': '1/np.tan', r'\basin\b': 'np.arcsin', r'\bacos\b': 'np.arccos',
        r'\batan\b': 'np.arctan', r'\bacsc\b': '1/np.arcsin', r'\basec\b': '1/np.arccos',
        r'\bacot\b': '1/np.arctan', r'\bsinh\b': 'np.sinh', r'\bcosh\b': 'np.cosh',
        r'\btanh\b': 'np.tanh', r'\bcsch\b': '1/np.sinh', r'\bsech\b': '1/np.cosh',
        r'\bcoth\b': '1/np.tanh', r'\basinh\b': 'np.arcsinh', r'\bacosh\b': 'np.arccosh',
        r'\batanh\b': 'np.arctanh', r'\babs\b': 'np.abs', r'\blog\b': 'np.log10', r'\bln\b': 'np.log'
    }

    def evaluate(self, equation, df_test, custom_functions=None):
        # Подготовка локальных переменных для eval (быстрее чем globals)
        eval_context = {col: df_test[col].values for col in df_test.columns}
        eval_context['np'] = np

        if custom_functions:
            eval_context.update(custom_functions)

        # Применяем замены
        for old, new in self.REPLACEMENTS.items():
            equation = re.sub(old, new, equation)

        try:
            # Передаем eval_context как локальный словарь
            p = eval(equation, {"__builtins__": {}}, eval_context)
        except Exception as e:
            print(f"Error evaluating equation: {e}")
            return None, equation

        return p, equation
