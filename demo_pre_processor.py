import pandas as pd
import numpy as np
import datetime


DAYS_IN_A_WEEK = 7.0
DAYS_IN_A_MONTH = 30.5


class MyPreprocessor:
    def __init__(self):
        pass

    def fit_transform(self, df):
        self.special_trans1 = CalendarUpdatedTrans(never_handle_type="mean")
        df = self.special_trans1.fit_transform(df)
        self.special_trans2 = HostResponseTrans(nan_handle_type="mean")
        df = self.special_trans2.fit_transform(df)
        self.special_trans3 = HostSinceTrans()
        df = self.special_trans3.fit_transform(df)
        self.special_trans4 = HostVerificationsTrans()
        df = self.special_trans4.fit_transform(df)
        self.special_trans5 = SquareFeetTrans(nan_handle_type="median")
        df = self.special_trans5.fit_transform(df)

        # These columns were found to be garbage by some validation.
        self.garbage_cols = ['listing_id', 'host_acceptance_rate', 'host_total_listings_count', 'picture_url',
                             'country_code', 'calendar_updated_or_not', 'host_verification_sent_id',
                             'host_verification_zhima_selfie', 'host_has_profile_pic',
                             'host_verification_manual_online','host_verification_Unknown',
                             'host_verification_photographer', 'host_verification_None',
                             'host_verification_weibo', 'is_business_travel_ready', 'has_availability',
                             'host_verification_sesame_offline']

        df.drop(self.garbage_cols, axis=1, inplace=True)

        return df

    def transform(self, df):
        df = self.special_trans1.transform(df)
        df = self.special_trans2.transform(df)
        df = self.special_trans3.transform(df)
        df = self.special_trans4.transform(df)
        df = self.special_trans5.transform(df)

        df.drop(self.garbage_cols, axis=1, inplace=True)

        return df


class HostVerificationsTrans:
    def __init__(self, lack_category="Unknown"):
        self.lack_category = lack_category

    def fit_transform(self, train_df):
        host_ver = train_df["host_verifications"].values
        num_samples = len(host_ver)
        lines = self._create_dictionary_and_lines(host_ver)
        features = self._create_features(num_samples, lines)
        train_df = self._fit_df(train_df, features)
        return train_df

    def transform(self, test_df):
        host_ver = test_df["host_verifications"].values
        num_samples = len(host_ver)
        lines = self._create_lines(host_ver)
        features = self._create_features(num_samples, lines)
        test_df = self._fit_df(test_df, features)
        return test_df

    def _create_dictionary_and_lines(self, host_ver):
        self.dictionary = set()
        lines = []
        for i, line in enumerate(host_ver):
            line = line.replace(" ", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("'", "")
            line = line.split(",")

            if line == [""]:
                line = [self.lack_category]

            lines.append(line)

            for word in line:
                self.dictionary.add(word)

        return lines

    def _create_lines(self, host_ver):
        lines = []
        for i, line in enumerate(host_ver):
            line = line.replace(" ", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("'", "")
            line = line.split(",")

            if line == [""]:
                line = [self.lack_category]

            lines.append(line)

        return lines

    def _create_features(self, num_samples, lines):
        # create a dict of numpy arrays for dictionary
        features = {}
        for word in self.dictionary:
            features[word] = np.zeros(num_samples, dtype=np.object)

        # one-hot-encoding
        for i, line in enumerate(lines):
            for word in self.dictionary:
                if word in line:
                    features[word][i] = "t"
                else:
                    features[word][i] = "f"

        return features

    def _fit_df(self, df, features):
        # create new column
        for word in self.dictionary:
            column_name = "host_verification_" + word
            df[column_name] = features[word]

        # drop old column
        df.drop("host_verifications", axis=1, inplace=True)

        return df


class SquareFeetTrans:
    def __init__(self, nan_handle_type="mean", nan_static_val=0):
        if nan_handle_type in ["mean", "static", "min", "median"]:
            self.nan_handle_type = nan_handle_type
        else:
            raise ValueError("invalid nan_handle_type (must be one of 'mean', 'static', 'min', 'median')")

        self.never_static_val = nan_static_val
        self.nan_val = 0

    def fit_transform(self, df):
        nan_indices = df.square_feet.isnull().values
        square_feet = df.square_feet.values

        not_nan_indices = nan_indices.__invert__()
        if self.nan_handle_type == "mean":
            self.nan_val = square_feet[not_nan_indices].mean()
        elif self.nan_handle_type == "min":
            self.nan_val = square_feet[not_nan_indices].min()
        elif self.nan_handle_type == "min":
            self.nan_val = square_feet[not_nan_indices].median()
        elif self.nan_handle_type == "static":
            self.nan_val = self.never_static_val
        square_feet[nan_indices] = self.nan_val

        category = np.zeros(len(nan_indices), dtype=np.object)
        category[not_nan_indices] = "t"
        category[nan_indices] = "f"

        df.square_feet = square_feet
        df["square_feet_or_not"] = category

        return df

    def transform(self, df):
        nan_indices = df.square_feet.isnull().values
        square_feet = df.square_feet.values

        square_feet[nan_indices] = self.nan_val

        category = np.zeros(len(nan_indices), dtype=np.object)
        not_nan_indices = nan_indices.__invert__()
        category[not_nan_indices] = "t"
        category[nan_indices] = "f"

        df.square_feet = square_feet
        df["square_feet_or_not"] = category

        return df


class CalendarUpdatedTrans:
    """
    #######################
    ###  handling rule  ###
    #######################
                                        days, category
    specials:
        "today"                    ->   0.0, t
        "yesterday"                ->   1.0, t
        "1 week ago"               ->   7.0, t
        "a week ago"               ->   7.0, t
        "never"                    ->   hyper_param * DAYS_IN_A_MONTH, f

    generals:
        str(num) + " days ago"     ->   num, t
        str(num) + " weeks ago"    ->   num * DAYS_IN_A_WEEK, t
        str(num) + " months ago"   ->   num * DAYS_IN_A_MONTH, t
    """
    def __init__(self, never_handle_type="mean", never_static_coeff=101):
        if never_handle_type in ["mean", "max", "static"]:
            self.never_handle_type = never_handle_type
        else:
            raise ValueError
        self.never_static_coeff = never_static_coeff
        self.never_val = 0.0

    def fit_transform(self, train_df):
        calendar_updated = train_df["calendar_updated"].values
        days, category = self._get_features(calendar_updated)

        indices_f = category == "f"
        indices_t = indices_f.__invert__()
        if self.never_handle_type == "mean":
            self.never_val = days[indices_t].mean()
            days[indices_f] = self.never_val
        elif self.never_handle_type == "max":
            self.never_val = days[indices_t].max()
            days[indices_f] = self.never_val

        train_df = train_df.assign(calendar_updated_days=0.)
        train_df.loc[:, "calendar_updated_days"] = days
        train_df = train_df.assign(calendar_updated_or_not="temp")
        train_df.loc[:, "calendar_updated_or_not"] = category
        train_df = train_df.drop("calendar_updated", axis=1)

        return train_df

    def transform(self, test_df):
        calendar_updated = test_df["calendar_updated"].values
        days, category = self._get_features(calendar_updated)

        test_df = test_df.assign(calendar_updated_days=0.)
        test_df.loc[:, "calendar_updated_days"] = days
        test_df = test_df.assign(calendar_updated_or_not="temp")
        test_df.loc[:, "calendar_updated_or_not"] = category
        test_df = test_df.drop("calendar_updated", axis=1)

        return test_df

    def _get_features(self, calendar_updated):
        num_samples = len(calendar_updated)
        days = np.zeros(num_samples, dtype=np.float)
        category = np.zeros(num_samples, dtype=np.object)

        for i, line in enumerate(calendar_updated):

            if "days ago" in line:
                num = float(line.split(" ")[0])
                days[i] = num
                category[i] = "t"
            elif "weeks ago" in line:
                num = float(line.split(" ")[0])
                days[i] = num * DAYS_IN_A_WEEK
                category[i] = "t"
            elif "months ago" in line:
                num = float(line.split(" ")[0])
                days[i] = num * DAYS_IN_A_MONTH
                category[i] = "t"
            elif line == "today":
                days[i] = 0.0
                category[i] = "t"
            elif line == "yesterday":
                days[i] = 1.0
                category[i] = "t"
            elif line == "1 week ago" or line == "a week ago":
                days[i] = 7.0
                category[i] = "t"
            elif line == "never":
                if self.never_handle_type == "static":
                    days[i] = self.never_static_coeff * DAYS_IN_A_MONTH
                else:
                    days[i] = self.never_val
                category[i] = "f"
            else:
                print("!!! warning: unexpected value (index: %d, str: %s) !!!" % (i, line))

        return days, category


class HostResponseTrans:
    """
    #######################
    ###  handling rule  ###
    #######################
                             hours, rate, category
    "within an hour"     ->  1.0, rate, t
    "within a few hours" ->  2.0, rate, t
    "within a day"       ->  3.0, rate, t
    "a few days or more" ->  4.0, rate, t
    nan                  ->  mean, mean, f
    """
    def __init__(self, nan_handle_type="mean"):
        if nan_handle_type in ["mean", "min"]:
            self.nan_handle_type = nan_handle_type
        else:
            raise ValueError

        self.nan_val_hours = 0.0
        self.nan_val_rate = 0.0

    def fit_transform(self, train_df):
        nan_indices = train_df["host_response_time"].isnull().values
        host_response_time = train_df["host_response_time"].values
        host_response_rate = train_df["host_response_rate"].values

        hours, rates, category = self._get_features(host_response_time,
                                                    host_response_rate,
                                                    nan_indices)

        not_nan_indices = nan_indices.__invert__()
        if self.nan_handle_type == "mean":
            self.nan_val_hours = hours[not_nan_indices].mean()
            self.nan_val_rate = rates[not_nan_indices].mean()
        elif self.nan_handle_type == "mim":
            self.nan_val_hours = hours[not_nan_indices].min()
            self.nan_val_rate = rates[not_nan_indices].min()
        hours[nan_indices] = self.nan_val_hours
        rates[nan_indices] = self.nan_val_rate

        train_df["host_response_hours"] = hours
        train_df["host_response_rate_new"] = rates
        train_df["host_response_category"] = category
        train_df = train_df.drop("host_response_time", axis=1)
        train_df = train_df.drop("host_response_rate", axis=1)

        return train_df

    def transform(self, test_df):
        nan_indices = test_df["host_response_time"].isnull().values
        host_response_time = test_df["host_response_time"].values
        host_response_rate = test_df["host_response_rate"].values

        hours, rates, category = self._get_features(host_response_time,
                                                    host_response_rate,
                                                    nan_indices)

        test_df["host_response_hours"] = hours
        test_df["host_response_rate_new"] = rates
        test_df["host_response_category"] = category
        test_df = test_df.drop("host_response_time", axis=1)
        test_df = test_df.drop("host_response_rate", axis=1)

        return test_df

    def _get_features(self, host_response_time,
                      host_response_rate,
                      nan_indices):
        num_samples = len(host_response_time)
        hours = np.zeros(num_samples, dtype=np.float)
        rates = np.zeros(num_samples, dtype=np.float)
        category = np.zeros(num_samples, dtype=np.object)

        for i in range(num_samples):

            if nan_indices[i]:
                hours[i] = self.nan_val_hours
                rates[i] = self.nan_val_rate
                category[i] = "f"
                continue

            category[i] = "t"

            time = host_response_time[i]
            if time == "within an hour":
                hours[i] = 1.0
            elif time == "within a few hours":
                hours[i] = 3.0
            elif time == "within a day":
                hours[i] = 24.0
            elif time == "a few days or more":
                hours[i] = 120.0
            else:
                print("!!! warning: unexpected value (index: %d, str: %s) !!!" % (i, time))

            rates[i] = float(host_response_rate[i].replace("%", ""))

        return hours, rates, category


class HostSinceTrans:
    def __init__(self):
        pass

    def fit_transform(self, train_df):
        host_since = train_df["host_since"].values
        dates, nan_indices = self._host_since_to_dates(host_since)
        self.reference_date = dates[0]      # param
        day_diff = self._dates_to_day_diff(dates, self.reference_date)
        self.day_diff_mean = day_diff.sum() / (len(day_diff) - len(nan_indices))     # param
        day_diff[nan_indices] = self.day_diff_mean
        train_df["day_diff"] = day_diff
        train_df = train_df.drop("host_since", axis=1)
        return train_df

    def transform(self, test_df):
        host_since = test_df["host_since"].values
        dates, nan_indices = self._host_since_to_dates(host_since)
        day_diff = self._dates_to_day_diff(dates, self.reference_date)
        day_diff[nan_indices] = self.day_diff_mean
        test_df["day_diff"] = day_diff
        test_df = test_df.drop("host_since", axis=1)
        return test_df

    def _host_since_to_dates(self, host_since):
        dates = []
        nan_indices = []
        for i in range(len(host_since)):
            if host_since[i] != host_since[i]:
                dates.append(0.0)
                nan_indices.append(i)
            else:
                splitted_date_str = host_since[i].split("-")
                dates.append(
                    datetime.date(year=int(splitted_date_str[0]),
                                  month=int(splitted_date_str[1]),
                                  day=int(splitted_date_str[2]))
                )

        return dates, nan_indices

    def _dates_to_day_diff(self, dates, reference_date):
        day_diff = []
        for i in range(len(dates)):
            if type(dates[i]) == datetime.date:
                day_diff.append((reference_date - dates[i]).days)
            else:
                day_diff.append(0.0)

        day_diff = np.array(day_diff).astype(np.float)

        return day_diff


