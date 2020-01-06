import pandas as pd
import numpy as np


class TransformDF2Numpy:
    def __init__(self, objective_col, y_zscore=True):
        self.objective_col = objective_col if type(objective_col) == str else ValueError("objective_col must be str")
        self.y_zscore_flag = y_zscore

    def fit_transform(self, df, min_category_count=0.):
        y = df[self.objective_col].values.copy()
        if self.y_zscore_flag:
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

        # transforms for all columns:
        # ・storing information of columns (categorical or numerical, and number of unique categories)
        # ・factorization of categorical variables
        # ・zscore normalization of numerical variables
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
                trans = Factorizer()
                df, self.variable_information = trans.fit_transform(df, col, self.variable_information,
                                                                    min_category_count)
                self.transforms.append(trans)
                categorical_transform_index.append(i)

            elif (num_uniques == 2) and (not is_numeric):
                trans = BinaryFactorizer()
                df, self.variable_information = trans.fit_transform(df, col, self.variable_information)
                self.transforms.append(trans)
                numerical_transform_index.append(i)

            elif is_numeric:
                trans = NumericalHandler()
                if col == "host_listing_count":
                    df, self.variable_information = trans.fit_transform(df, col, self.variable_information,
                                                                        na_type="median")
                else:
                    df, self.variable_information = trans.fit_transform(df, col, self.variable_information)
                self.transforms.append(trans)
                numerical_transform_index.append(i)

            else:
                message = "something wrong with column: " + col
                raise Exception(message)

        self.variable_information["variables"] = self.variable_information["categorical_variables"]\
                                                 + self.variable_information["numerical_variables"]
        self.variable_information["transform_index"] = categorical_transform_index + numerical_transform_index

        self.num_categorical = len(self.variable_information["categorical_variables"])
        self.num_numerical = len(self.variable_information["numerical_variables"])

        x = self._df_to_numpy(df)

        return x, y

    def transform(self, df):
        y = df[self.objective_col].values.copy()
        if self.y_zscore_flag:
            y = (y - self.y_mean) / self.y_std

        for i, col in enumerate(df.columns):
            df = self.transforms[i].transform(df, col)

        x = self._df_to_numpy(df)

        return x, y

    def _df_to_numpy(self, df):
        x_categorical = df[self.variable_information["categorical_variables"]].values
        x_numerical = df[self.variable_information["numerical_variables"]].values
        return np.concatenate([x_categorical, x_numerical], axis=1)

    def get_transform(self, index_or_colname):
        if type(index_or_colname) == int:
            return self.transforms[self.variable_information["transform_index"][index_or_colname]]
        elif type(index_or_colname) == str:
            index = self.variable_information["variables"].index(index_or_colname)
            return self.transforms[self.variable_information["transform_index"][index]]
        else:
            raise ValueError("Input must be a index (int) or a name of variable (str)")



class CategoryThreshold:
    def __init__(self):
        pass

    def fit_transform(self, df, col_name, acum_ratio=0.99, min_count=5):
        val_cnt = df[col_name].value_counts()

        drop_target_series = val_cnt.cumsum() / val_cnt.sum() > acum_ratio
        drop_target_series_count = val_cnt < min_count
        drop_target_series = drop_target_series | drop_target_series_count

        self.drop_targets = drop_target_series[drop_target_series].index
        df[col_name].replace(self.drop_targets, "category_thresholded", inplace=True)
        print("category thresholded (", col_name, ")", end="")
        print(", dropped categories: ", len(self.drop_targets))
        return df

    def transform(self, df, col_name):
        df[col_name].replace(self.drop_targets, "category_thresholded", inplace=True)
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
    def __init__(self):
        pass

    def fit_transform(self, df, col_name, variable_info, min_category_count):
        self.col_name = col_name

        self.ct = CategoryThreshold()
        df = self.ct.fit_transform(df, col_name, acum_ratio=1., min_count=min_category_count)
        df[col_name], self.dictionary = df[col_name].factorize()
        if -1 in df[col_name].values:
            self.fill_value = df[col_name].values.max() + 1.
            df.loc[df[col_name] == -1, col_name] = self.fill_value
        else:
            self.fill_value = df[col_name].value_counts().index[0]

        variable_info["categorical_variables"].append(col_name)
        variable_info["categorical_uniques"].append(df[col_name].nunique())

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
    def __init__(self):
        pass

    def fit_transform(self, df, col_name, variable_info):
        self.col_name = col_name

        df[col_name], self.dictionary = df[col_name].factorize()
        self.fill_value = df[col_name].value_counts().index[0]

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
        df[col_name] = (df[col_name].values - self.mean) / self.std

        return df


class NumericalHandler:
    def __init__(self):
        pass

    def fit_transform(self, df, col_name, variable_info, na_type="mean"):
        self.col_name = col_name

        if na_type == "mean":
            self.na_value = df[col_name].mean()
        elif na_type == "median":
            self.na_value = df[col_name].median()
        else:
            raise ValueError("invalid na_type")
        df[col_name].fillna(self.na_value, inplace=True)
        self.mean = df[col_name].mean()
        self.std = df[col_name].std()
        df[col_name] = (df[col_name].values - self.mean) / self.std

        variable_info["numerical_variables"].append(col_name)

        return df, variable_info

    def transform(self, df, col_name):
        if col_name != self.col_name:
            raise ValueError("Could not transform. DataFrame construction is wrong.")

        df[col_name].fillna(self.na_value, inplace=True)
        df[col_name] = (df[col_name].values - self.mean) / self.std

        return df


# test
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from demo_pre_processor import MyPreprocessor

    csv_dir = "./demodata.csv"
    df = pd.read_csv(csv_dir)
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=10)

    p = MyPreprocessor()
    processed_df_train = p.fit_transform(df_train)
    processed_df_val = p.transform(df_val)

    t = TransformDF2Numpy(y_zscore=True)
    x_train, y_train = t.fit_transform(processed_df_train, min_category_count=3)
    x_val, y_val = t.transform(processed_df_val)

    for i, variable in enumerate(t.variable_information["variables"]):
        variable_type = "categorical" if i < t.num_categorical else "numerical"
        print("(variable index %d, %s) " % (i, variable_type), end="")
        print(variable, end="")
        print("  (transformer: ", type(t.get_transform(i)), ")")



