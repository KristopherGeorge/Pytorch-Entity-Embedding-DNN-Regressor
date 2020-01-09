import pandas as pd
import numpy as np
import warnings


def message_when_fitted(info):
    print("Transformer (TransformDF2Numpy) fitted.")
    print("Number of the categorical variables:", len(info["categorical_variables"]))
    print("Number of the numerical variables:", len(info["numerical_variables"]))
    print("------------------------------------------------")


class TransformDF2Numpy:
    def __init__(self, objective_col,
                 objective_scaling=False,
                 numerical_scaling=False,
                 min_category_count=0.,
                 scaling_robustness_factor=0.,
                 fillnan_robustness_factor=0.):
        self.objective_col = objective_col if type(objective_col) == str else ValueError("objective_col must be str")
        self.objective_scaling_flag = objective_scaling
        self.numerical_scaling = numerical_scaling
        self.min_category_count = min_category_count
        self.scaling_robustness_factor = scaling_robustness_factor
        self.fillnan_robustness_factor = fillnan_robustness_factor

    def fit_transform(self, df):
        y = df[self.objective_col].values.copy()
        if self.objective_scaling_flag:
            self.y_mean = y.mean()
            self.y_std = y.std()
            y = (y - self.y_mean) / self.y_std

        # information of variables
        self.variable_information = {
            "variables": None,
            "transform_index": None,
            "categorical_variables": [],
            "numerical_variables": [],
            "categorical_uniques": []
        }

        self.transforms = []
        categorical_transform_index = []
        numerical_transform_index = []
        for i, col in enumerate(df.columns):
            num_uniques = df[col].unique().size
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            if (col == self.objective_col) or (num_uniques == 1):
                trans = Dropper()
                df, self.variable_information = trans.fit_transform(df, col, self.variable_information)
                self.transforms.append(trans)

            elif (num_uniques > 2) and (not is_numeric):
                trans = Factorizer(self.min_category_count)
                df, self.variable_information = trans.fit_transform(df, col, self.variable_information)
                self.transforms.append(trans)
                categorical_transform_index.append(i)

            elif (num_uniques == 2) and (not is_numeric):
                trans = BinaryFactorizer(self.numerical_scaling)
                df, self.variable_information = trans.fit_transform(df, col, self.variable_information)
                self.transforms.append(trans)
                numerical_transform_index.append(i)

            elif is_numeric:
                trans = NumericalHandler(self.numerical_scaling, self.scaling_robustness_factor,
                                         self.fillnan_robustness_factor)
                df, self.variable_information = trans.fit_transform(df, col, self.variable_information)
                self.transforms.append(trans)
                numerical_transform_index.append(i)

            else:
                message = "something wrong with column: " + col
                raise Exception(message)

        self.variable_information["variables"] = self.variable_information["categorical_variables"]\
                                                 + self.variable_information["numerical_variables"]
        self.variable_information["transform_index"] = categorical_transform_index + numerical_transform_index

        self.num_categoricals = len(self.variable_information["categorical_variables"])
        self.num_numericals = len(self.variable_information["numerical_variables"])

        x = self._df_to_numpy(df)

        message_when_fitted(self.variable_information)

        return x, y

    def transform(self, df):
        y = df[self.objective_col].values.copy()
        if self.objective_scaling_flag:
            y = (y - self.y_mean) / self.y_std

        for i, col in enumerate(df.columns):
            df = self.transforms[i].transform(df, col)

        x = self._df_to_numpy(df)

        return x, y

    def _df_to_numpy(self, df):
        x_categorical = df[self.variable_information["categorical_variables"]].values
        x_numerical = df[self.variable_information["numerical_variables"]].values
        return np.concatenate([x_categorical, x_numerical], axis=1)

    def _get_transform(self, index_or_colname):
        if type(index_or_colname) in [int, np.int, np.int8, np.int16, np.int32, np.int64]:
            return self.transforms[self.variable_information["transform_index"][index_or_colname]]
        elif type(index_or_colname) == str:
            index = self.variable_information["variables"].index(index_or_colname)
            return self.transforms[self.variable_information["transform_index"][index]]
        else:
            raise ValueError("Input must be an index (int) or a name of variable (str)")

    def dictionary(self, index_or_colname):
        trans = self._get_transform(index_or_colname)
        if type(trans) == Factorizer:
            return trans.dictionary
        else:
            raise ValueError("Specified variable is not categorical.")

    def dropped_categories(self, index_or_colname):
        trans = self._get_transform(index_or_colname)
        if type(trans) == Factorizer:
            return trans.ct.drop_targets
        else:
            raise ValueError("Specified variable is not categorical.")

    def categorical_uniques(self, index_or_colname=None):
        if index_or_colname is not None:
            trans = self._get_transform(index_or_colname)
            if type(trans) == Factorizer:
                return trans.num_uniques
            else:
                raise ValueError("Specified variable is not categorical.")
        else:
            return self.variable_information["categorical_uniques"]

    def variables(self):
        return self.variable_information["variables"]

    def categoricals(self):
        return self.variable_information["categorical_variables"]

    def numericals(self):
        return self.variable_information["numerical_variables"]


class CategoryThreshold:
    def __init__(self):
        pass

    def fit_transform(self, df, col_name, min_count=5):
        val_cnt = df[col_name].value_counts()
        drop_target_series = val_cnt < min_count
        self.drop_targets = drop_target_series[drop_target_series].index

        df[col_name].replace(self.drop_targets, "dropped_category", inplace=True)
        if len(self.drop_targets) != 0:
            print("category thresholded in '%s', dropped categories: %d" % (col_name, len(self.drop_targets)))
        return df

    def transform(self, df, col_name):
        df[col_name].replace(self.drop_targets, "dropped_category", inplace=True)
        return df


class Dropper:
    def __init__(self):
        pass

    def fit_transform(self, df, col_name, variable_info):
        self.col_name = col_name
        return df, variable_info

    def transform(self, df, col_name):
        if col_name != self.col_name:
            raise ValueError("Could not transform. DataFrame construction is wrong.")
        return df


class Factorizer:
    def __init__(self, min_category_count):
        self.min_category_count = min_category_count

    def fit_transform(self, df, col_name, variable_info):
        self.col_name = col_name

        self.ct = CategoryThreshold()
        df = self.ct.fit_transform(df, col_name, min_count=self.min_category_count)
        df[col_name], self.dictionary = df[col_name].factorize()
        if -1 in df[col_name].values:
            self.fill_value = df[col_name].values.max() + 1.
            df.loc[df[col_name] == -1, col_name] = self.fill_value
        else:
            self.fill_value = df[col_name].value_counts().index[0]

        variable_info["categorical_variables"].append(col_name)
        self.num_uniques = df[col_name].nunique()
        variable_info["categorical_uniques"].append(self.num_uniques)

        return df, variable_info

    def transform(self, df, col_name):
        if col_name != self.col_name:
            raise ValueError("Could not transform. DataFrame construction is wrong.")
        df = self.ct.transform(df, col_name)

        df[col_name] = self.dictionary.get_indexer(df[col_name])
        if -1 in df[col_name].values:
            df.loc[df[col_name] == -1, col_name] = self.fill_value

        return df


class BinaryFactorizer:
    def __init__(self, scaling_flag):
        self.scaling_flag = scaling_flag

    def fit_transform(self, df, col_name, variable_info):
        self.col_name = col_name

        df[col_name], self.dictionary = df[col_name].factorize()
        self.fill_value = df[col_name].value_counts().index[0]

        if self.scaling_flag:
            self.mean = df[col_name].values.mean()
            self.std = df[col_name].values.std()
            df[col_name] = (df[col_name].values - self.mean) / self.std

        variable_info["numerical_variables"].append(col_name)

        return df, variable_info

    def transform(self, df, col_name):
        if col_name != self.col_name:
            raise ValueError("Could not transform. DataFrame construction is wrong.")

        df[col_name] = self.dictionary.get_indexer(df[col_name])
        if -1 in df[col_name].values:
            df.loc[df[col_name] == -1, col_name] = self.fill_value

        if self.scaling_flag:
            df[col_name] = (df[col_name].values - self.mean) / self.std

        return df


def get_fillnan(values, fillnan_robustness_factor):
    values = values[~np.isnan(values)]
    values = np.sort(values)
    start_index = int(len(values) / 2 * fillnan_robustness_factor)  # robustness_factorは片側
    gorl_index = int(len(values) - start_index - 1)
    nan_value = values[start_index:gorl_index].mean()
    return nan_value


def get_mean_std(values, scaling_robustness_factor, col_name):
    values = np.sort(values)
    start_index = int(len(values) / 2 * scaling_robustness_factor)  # robustness_factorは片側
    gorl_index = int(len(values) - start_index - 1)
    std = values[start_index:gorl_index].std() + 0.000001
    if std == 0.000001:
        message = "Warning: Robust scaling of the variable:'%s' was failed due to infinite std appeared." % col_name\
                  + " The mean and std will be calculated by all values instead."
        warnings.warn(message)
        std = values.std() + 0.000001
        mean = values.mean()
        return mean, std
    else:
        mean = values[start_index:gorl_index].mean()
        return mean, std


class NumericalHandler:
    def __init__(self, scaling_flag, scaling_robustness_factor, fillnan_robustness_factor):
        self.scaling_flag = scaling_flag
        self.scaling_robustness_factor = scaling_robustness_factor
        self.fillnan_robustness_factor = fillnan_robustness_factor

    def fit_transform(self, df, col_name, variable_info):
        self.col_name = col_name

        self.nan_value = get_fillnan(df[col_name].values, self.fillnan_robustness_factor)
        nan_count = (df[col_name].isnull()).sum()
        if nan_count:
            print("nan value filled in '%s', (filled rows: %d, value: %f)" % (col_name, nan_count, self.nan_value))
            df[col_name].fillna(self.nan_value, inplace=True)

        if self.scaling_flag:
            self.mean, self.std = get_mean_std(df[col_name].values, self.scaling_robustness_factor, col_name)
            df[col_name] = (df[col_name].values - self.mean) / self.std

        variable_info["numerical_variables"].append(col_name)

        return df, variable_info

    def transform(self, df, col_name):
        if col_name != self.col_name:
            raise ValueError("Could not transform. DataFrame construction is wrong.")

        df[col_name].fillna(self.nan_value, inplace=True)

        if self.scaling_flag:
            df[col_name] = (df[col_name].values - self.mean) / self.std

        return df


# test
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from demo_pre_processor import MyPreprocessor
    import matplotlib.pyplot as plt

    csv_dir = "./demodata.csv"
    df = pd.read_csv(csv_dir)
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=10)

    p = MyPreprocessor()
    processed_df_train = p.fit_transform(df_train)
    processed_df_val = p.transform(df_val)

    t = TransformDF2Numpy('price', objective_scaling=True, numerical_scaling=True, min_category_count=3,
                          scaling_robustness_factor=0.01, fillnan_robustness_factor=0.0001)
    x_train, y_train = t.fit_transform(processed_df_train)
    x_val, y_val = t.transform(processed_df_val)

    print()
    print("all variables:")
    print(t.variables())
    print()

    print("categorical variables:")
    print(t.categoricals())
    print()

    print("numerical variables:")
    print(t.numericals())
    print()

    print("number of the unique categories of the categorical variables:")
    print(t.categorical_uniques())
    print()

    print("Unique categories of a specific variable, the dictionary, and the dropped categories:")
    # index_or_column_name = 4
    index_or_column_name = 'property_type'
    print(t.categorical_uniques(index_or_column_name))
    print(t.dictionary(index_or_column_name))
    print(t.dropped_categories(index_or_column_name))



