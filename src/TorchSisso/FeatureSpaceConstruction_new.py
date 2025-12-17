'''
##############################################################################################

Importing the required libraries

##############################################################################################
'''
from itertools import combinations as py_combinations
import torch
import pandas as pd
import numpy as np
import warnings
import itertools
import time
from . import combinations_construction
from . import unary_construction


class feature_space_construction:

    '''
    ##############################################################################################################

    Define global variables like number of operators and the input data frame and the operator set given

    ##############################################################################################################
    '''

    def __init__(self, operators, df, no_of_operators=None, device='cpu', initial_screening=None, custom_unary_functions=None, custom_binary_functions=None):

        print(
            f'************************************************ Starting Feature Space Construction in {device} ************************************************ \n')
        print('\n')
        '''
    ###########################################################################################

    no_of_operators - defines the presence of operators (binary or unary) in the expanded features space

    For example: if no_of_operators = 2 then the space will be limited to formation of features with 3 operators (x1+x2)/x3 or exp(x1+x2)

    ###########################################################################################
    '''
        self.no_of_operators = no_of_operators

        self.df = df
        '''
    ###########################################################################################

    operators [list type]: Defines the mathematical operators needs to be used in the feature expansion

    Please look at the README.md for type of mathematical operators allowed

    ###########################################################################################
    '''
        self.operators = operators

        self.device = torch.device(device)

        # Filter the dataframe by removing the categorical datatypes and zero variance feature variables

        self.df = self.df.select_dtypes(
            include=['float64', 'int64', 'int32', 'float32'])

        # Compute the variance of each column
        variance = self.df.var()

        # Get the names of the zero variance columns
        zero_var_cols = variance[variance == 0].index

        # Drop the zero variance columns from the dataframe
        self.df = self.df.drop(zero_var_cols, axis=1)

        # Pop out the Targer variable of the problem and convert to tensor
        self.df.rename(
            columns={f'{self.df.columns[0]}': 'Target'}, inplace=True)

        self.Target_column = torch.tensor(
            self.df.pop('Target')).to(self.device)

        if initial_screening != None:

            self.screening = initial_screening[0]

            self.quantile = initial_screening[1]

            self.df = self.feature_space_screening(self.df)

        # Create the feature values tensor
        self.df_feature_values = torch.tensor(self.df.values).to(self.device)

        self.columns = self.df.columns.tolist()

        # Create a dataframe for appending new datavalues
        self.new_features_values = pd.DataFrame()

        # Creating empty tensor and list for single operators (Unary operators)
        self.feature_values_unary = torch.empty(
            self.df.shape[0], 0).to(self.device)

        self.feature_names_unary = []

        # creating empty tensor and list for combinations (Binary Operators)
        self.feature_values_binary = torch.empty(
            self.df.shape[0], 0).to(self.device)

        self.feature_names_binary = []

        self.custom_unary_functions = custom_unary_functions

        self.custom_binary_functions = custom_binary_functions

        self.combination_cache = {}

    '''
  ###############################################################################################################

  Construct all the features that can be constructed using the single operators like log, exp, sqrt etc..

  ###############################################################################################################
  '''

    def feature_space_screening(self, df_sub):

        from sklearn.feature_selection import mutual_info_regression
        from scipy.stats import spearmanr

        if self.screening == 'spearman':
            spear = spearmanr(df_sub.to_numpy(), self.Target_column, axis=0)
            screen1 = abs(spear.statistic)
            if screen1.ndim > 1:
                screen1 = screen1[:-1, -1]
        elif self.screening == 'mi':
            screen1 = mutual_info_regression(
                df_sub.to_numpy(), self.Target_column.numpy())

        df_screening = pd.DataFrame()
        df_screening['Feature variables'] = df_sub.columns
        df_screening['screen1'] = screen1
        df_screening = df_screening.sort_values(
            by='screen1', ascending=False).reset_index(drop=True)
        quantile_screen = df_screening.screen1.quantile(self.quantile)

        filtered_df = df_screening[(
            df_screening.screen1 > quantile_screen)].reset_index(drop=True)

        if filtered_df.shape[0] == 0:
            filtered_df = df_screening[:int(df_sub.shape[1]/2)]

        df_screening1 = df_sub.loc[:,
                                   filtered_df['Feature variables'].tolist()]

        return df_screening1

    def single_variable(self, operators_set, i):

        # Looping over operators set to get the new features/predictor variables

        if len(operators_set) == 0 and self.custom_unary_functions != None:

            self.feature_values_11 = torch.empty(
                self.df.shape[0], 0).to(self.device)

            feature_names_12 = []

            applier = unary_construction.FunctionApplier()

            self.feature_values_11, text_representations = applier.construct_function(
                self.custom_unary_functions, self.df_feature_values)

            for i in range(len(text_representations)):

                feature_names_12.extend(list(map(lambda x: str(
                    text_representations[i][0]) + x + str(text_representations[i][1]), self.columns)))

            self.feature_values_unary = torch.cat(
                (self.feature_values_unary, self.feature_values_11), dim=1)

            self.feature_names_unary.extend(feature_names_12)

            del self.feature_values_11, feature_names_12

        for op in operators_set:

            self.feature_values_11 = torch.empty(
                self.df.shape[0], 0).to(self.device)
            feature_names_12 = []

            # Performs the exponential transformation of the given feature space

            if self.custom_unary_functions != None:

                applier = unary_construction.FunctionApplier()
                self.feature_values_11, text_representations = applier.construct_function(
                    self.custom_unary_functions, self.df_feature_values)

                for i in range(len(text_representations)):
                    feature_names_12.extend(list(map(lambda x: str(
                        text_representations[i][0]) + x + str(text_representations[i][1]), self.columns)))

            if op == 'exp':
                exp = torch.exp(self.df_feature_values)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, exp), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(exp(' + x + "))", self.columns)))

            # Performs the natural lograithmic transformation of the given feature space
            elif op == 'ln':

                ln = torch.log(self.df_feature_values)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, ln), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(ln('+x + "))", self.columns)))

            # Performs the lograithmic transformation of the given feature space
            elif op == 'log':
                log10 = torch.log10(self.df_feature_values)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, log10), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(log('+x + "))", self.columns)))

            # Performs the power transformations of the feature variables..

            elif "pow" in op:

                import re

                pattern = r'\(([^)]*)\)'
                matches = re.findall(pattern, op)
                op = eval(matches[0])

                transformation = torch.pow(self.df_feature_values, op)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, transformation), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '('+x + f")**{matches[0]}", self.columns)))

            # Performs the sine function transformation of the given feature space
            elif op == 'sin':
                sin = torch.sin(self.df_feature_values)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, sin), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(sin('+x + "))", self.columns)))

            # Performs the hyperbolic sine function transformation of the given feature space
            elif op == 'sinh':
                sin = torch.sinh(self.df_feature_values)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, sin), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(sinh('+x + "))", self.columns)))

            # Performs the cosine transformation of the given feature space
            elif op == 'cos':
                cos = torch.cos(self.df_feature_values)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, cos), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(cos('+x + "))", self.columns)))

            # Performs the hyperbolic cosine transformation of the given feature space
            elif op == 'cosh':
                cos = torch.cosh(self.df_feature_values)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, cos), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(cosh('+x + "))", self.columns)))

            # Performs the hyperbolic tan transformation of the given feature space
            elif op == 'tanh':
                cos = torch.tanh(self.df_feature_values)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, cos), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(tanh('+x + "))", self.columns)))

            # Performs the Inverse transformation of the given feature space
            elif op == '^-1':
                reciprocal = torch.reciprocal(self.df_feature_values)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, reciprocal), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(('+x + ")**(-1))", self.columns)))

            # Performs the Inverse exponential transformation of the given feature space
            elif op == 'exp(-1)':
                exp = torch.exp(self.df_feature_values)
                expreciprocal = torch.reciprocal(exp)
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, expreciprocal), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '(exp(-'+x + "))", self.columns)))

            elif op == '+1':
                add1 = self.df_feature_values + 1
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, add1), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '('+x + "+1)", self.columns)))

            elif op == '-1':
                sub1 = self.df_feature_values - 1
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, sub1), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '('+x + "-1)", self.columns)))

            elif op == '/2':
                div2 = self.df_feature_values/2
                self.feature_values_11 = torch.cat(
                    (self.feature_values_11, div2), dim=1)
                feature_names_12.extend(
                    list(map(lambda x: '('+x + "/2)", self.columns)))

            self.feature_values_unary = torch.cat(
                (self.feature_values_unary, self.feature_values_11), dim=1)

            self.feature_names_unary.extend(feature_names_12)

            del self.feature_values_11, feature_names_12

        # Check for empty lists
        if len(self.feature_names_unary) == 0:
            return self.feature_values_unary, self.feature_names_unary
        else:
            # create Boolean masks for NaN and Inf values

            nan_columns = torch.any(torch.isnan(
                self.feature_values_unary), dim=0)
            inf_columns = torch.any(torch.isinf(
                self.feature_values_unary), dim=0)
            nan_or_inf_columns = nan_columns | inf_columns

            # Remove columns from tensor
            self.feature_values_unary = self.feature_values_unary[:,
                                                                  ~nan_or_inf_columns]

            # Remove corresponding elements from list
            self.feature_names_unary = [elem for i, elem in enumerate(
                self.feature_names_unary) if not nan_or_inf_columns[i]]

            if self.no_of_operators != None:

                if i+1 != self.no_of_operators:

                    # Get the duplicate columns in the feature space created..
                    unique_columns, indices = torch.unique(
                        self.feature_values_unary, sorted=False, dim=1, return_inverse=True)

                    # Get the indices of the unique columns
                    unique_indices = indices.unique()

                    # Remove duplicate columns
                    self.feature_values_unary = self.feature_values_unary[:,
                                                                          unique_indices]

                    # Remove the corresponding elements from the list of feature names..
                    self.feature_names_unary = [
                        self.feature_names_unary[i] for i in unique_indices.tolist()]

            return self.feature_values_unary, self.feature_names_unary

    '''
  ################################################################################################

  Defining method to perform the combinations of the variables with the initial feature set
  ################################################################################################
  '''

    def combinations(self, operators_set, i):
        # 1. Строгое кеширование с неизменяемым ключом
        # Превращаем set в отсортированный tuple для уникальности ключа
        op_key = tuple(sorted(list(operators_set)))

        # Быстрая проверка наличия в кеше
        if op_key in self.combination_cache:
            if i in self.combination_cache[op_key]:
                cached_val = self.combination_cache[op_key][i]
                if cached_val is not None:
                    # Возвращаем копии, чтобы не испортить кеш извне
                    return cached_val[0].clone(), cached_val[1][:]
        else:
            self.combination_cache[op_key] = {}

        # Используем no_grad, так как здесь не нужны градиенты (экономит память и время)
        with torch.no_grad():
            new_features_list = []
            new_names_list = []

            # --- Блок Custom Binary Functions ---
            if len(operators_set) == 0 and self.custom_binary_functions is not None:
                constructor = combinations_construction.FeatureConstructor(
                    self.df_feature_values, self.columns)
                results, expressions = constructor.construct_generic_features(
                    self.custom_binary_functions)

                new_features_list.append(results)
                new_names_list.extend(expressions)

            # --- Блок Стандартных Операторов ---
            elif len(operators_set) > 0:
                # Предварительная подготовка индексов и данных
                # Получаем индексы всех пар (N_pairs, 2)
                # Важно: используем device для индексов, чтобы индексация шла на GPU
                num_features = self.df_feature_values.shape[1]
                comb_indices = torch.combinations(
                    torch.arange(num_features, device=self.device), 2)

                # Извлекаем левые и правые операнды сразу в формате (Samples, Pairs)
                # Это исключает необходимость в permute и transpose (.T) в дальнейшем
                left_ops = self.df_feature_values[:, comb_indices[:, 0]]
                right_ops = self.df_feature_values[:, comb_indices[:, 1]]

                # Предварительная генерация списка пар имен (CPU операция)
                # Делаем это один раз, а не внутри цикла по операторам
                col_pairs = list(py_combinations(self.columns, 2))

                for op in operators_set:
                    if op == '+':
                        # Операция сразу над всеми парами: (Samples, Pairs) + (Samples, Pairs)
                        res = left_ops + right_ops
                        new_features_list.append(res)
                        # Быстрая генерация имен без lambda
                        new_names_list.extend(
                            [f"({c1}+{c2})" for c1, c2 in col_pairs])

                    elif op == '-':
                        res = left_ops - right_ops
                        new_features_list.append(res)
                        new_names_list.extend(
                            [f"({c1}-{c2})" for c1, c2 in col_pairs])

                    elif op == '*':
                        res = left_ops * right_ops
                        new_features_list.append(res)
                        new_names_list.extend(
                            [f"({c1}*{c2})" for c1, c2 in col_pairs])

                    elif op == '/':
                        # Деление генерирует в 2 раза больше признаков (A/B и B/A)
                        div1 = left_ops / right_ops
                        div2 = right_ops / left_ops
                        new_features_list.append(div1)
                        new_features_list.append(div2)
                        new_names_list.extend(
                            [f"({c1}/{c2})" for c1, c2 in col_pairs])
                        new_names_list.extend(
                            [f"({c2}/{c1})" for c1, c2 in col_pairs])

                # Если есть кастомные функции вместе с операторами (редкий кейс, но поддержим)
                if self.custom_binary_functions is not None:
                    constructor = combinations_construction.FeatureConstructor(
                        self.df_feature_values, self.columns)
                    results, expressions = constructor.construct_generic_features(
                        self.custom_binary_functions)
                    new_features_list.append(results)
                    new_names_list.extend(expressions)

            # --- Сборка и Очистка ---

            # Если ничего не сгенерировали (пустой список)
            if not new_features_list:
                # Возвращаем текущее состояние (возможно, пустое)
                self.combination_cache[op_key][i] = (
                    self.feature_values_binary, self.feature_names_binary)
                return self.feature_values_binary, self.feature_names_binary

            # 2. Однократная конкатенация новых признаков
            new_features_tensor = torch.cat(new_features_list, dim=1)

            # Конкатенация с уже существующими бинарными признаками (накопление)
            # Обратите внимание: в оригинале self.feature_values_binary накапливается.
            combined_features = torch.cat(
                (self.feature_values_binary, new_features_tensor), dim=1)
            combined_names = self.feature_names_binary + new_names_list

            # 3. Очистка от NaN/Inf (Векторизовано)
            # Проверяем только combined_features.
            # Оптимизация: torch.any(dim=0) возвращает маску плохих колонок
            is_nan = torch.isnan(combined_features).any(dim=0)
            is_inf = torch.isinf(combined_features).any(dim=0)
            to_drop_mask = is_nan | is_inf  # Логическое ИЛИ

            if to_drop_mask.any():
                # Оставляем только "хорошие" колонки (~to_drop_mask)
                good_indices = (~to_drop_mask).nonzero(as_tuple=True)[0]
                combined_features = combined_features[:, good_indices]
                # Фильтруем имена (с помощью itertools.compress было бы быстрее для огромных списков,
                # но list comprehension здесь понятнее и достаточно быстр)
                # Нам нужно преобразовать маску тензора в список булевых значений для фильтрации имен
                keep_mask_list = (~to_drop_mask).tolist()
                combined_names = [name for name, keep in zip(
                    combined_names, keep_mask_list) if keep]

            # 4. Удаление дубликатов (Самая тяжелая операция)
            # Выполняем unique только если это не последняя итерация
            if self.operators is not None and (i + 1 != self.no_of_operators):
                # unique с dim=1 очень дорогая операция.
                # sorted=False может немного ускорить
                unique_vals, unique_indices = torch.unique(
                    combined_features, sorted=False, dim=1, return_inverse=True
                )
                # unique_indices возвращает индексы оригинального тензора, которые формируют unique вывод
                # Но torch.unique возвращает сами уникальные значения первым аргументом

                # Нам нужны только уникальные колонки.
                # torch.unique(dim=1) уже вернул unique_vals - это тензор без дубликатов.
                # Но нам нужны индексы, чтобы отфильтровать имена.

                # Трюк: чтобы получить индексы ПЕРВЫХ вхождений уникальных элементов (для фильтрации имен):
                # К сожалению, torch.unique не имеет return_index (как numpy).
                # Поэтому используем следующий подход:
                perm = torch.arange(unique_indices.size(
                    0), dtype=unique_indices.dtype, device=unique_indices.device)
                unique_indices_of_first_occurence = perm.new_empty(
                    unique_vals.size(1)).scatter_(0, unique_indices, perm)
                # Сортируем индексы, чтобы сохранить порядок признаков (опционально, но полезно)
                unique_indices_of_first_occurence, _ = unique_indices_of_first_occurence.sort()

                combined_features = combined_features[:,
                                                      unique_indices_of_first_occurence]
                combined_names = [combined_names[idx]
                                  for idx in unique_indices_of_first_occurence.tolist()]

            # Обновляем состояние класса
            self.feature_values_binary = combined_features
            self.feature_names_binary = combined_names

            # Сохраняем в кеш (клонируем тензор, имена копируем срезом)
            self.combination_cache[op_key][i] = (
                self.feature_values_binary.clone(), self.feature_names_binary[:])

            return self.feature_values_binary, self.feature_names_binary

    '''
  ##########################################################################################################

  Creating the space based on the given set of conditions

  ##########################################################################################################

  '''

    def feature_space(self):

        basic_operators = [
            op for op in self.operators if op in ['+', '-', '*', '/']]
        other_operators = [
            op for op in self.operators if op not in ['+', '-', '*', '/']]
        for i in range(1, self.no_of_operators):

            start_time = time.time()

            print(
                f'************************************************ Starting {i} level of feature expansion...************************************************ \n')

            # Performs the feature space expansion based on the binary operator set provided
            values, names = self.combinations(basic_operators, i)

            # Performs the feature space expansion based on the unary operator set provided
            values1, names1 = self.single_variable(other_operators, i)

            features_created = torch.cat((values, values1), dim=1)

            del values, values1

            names2 = names + names1

            del names, names1

            self.df_feature_values = torch.cat(
                (self.df_feature_values, features_created), dim=1)

            self.columns.extend(names2)

            del features_created, names2

            print(f'************************************************ {i} Feature Expansion Completed with feature space size:::',
                  self.df_feature_values.shape[1], '************************************************ \n')

            print('************************************************ Time taken to create the space is:::',
                  time.time()-start_time, ' Seconds...************************************************ \n')

        return self.df_feature_values, self.Target_column, self.columns
