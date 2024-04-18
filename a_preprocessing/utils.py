from sklearn import preprocessing

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def convert_timestamp_to_hour(timestamp):
    try:
        hour = int(timestamp.split("T")[-1].split(":")[0])
    except:
        result = "evening"
        print(f"Cannot decode hour, use {result}!")
        return result

    if hour >= 0 and hour < 6:
        result = "dawn"

    elif hour >= 6 and hour < 12:
        result = "morning"

    elif hour >= 12 and hour < 18:
        result = "afternoon"

    else:
        result = "evening"

    return result


def convert_timestamp_to_day(timestamp):
    try:
        day = int(timestamp.split("T")[0].split("-")[-1])
        index = day % 7
    except:
        index = 0
        print(f"Cannot decode day, use {WEEK[index]}!")

    return WEEK[index]


def split_timestamp_into_day_and_hour(data_df, column):
    data_df[f"{column}_day"] = data_df[column].apply(convert_timestamp_to_day)
    data_df[f"{column}_hour"] = data_df[column].apply(convert_timestamp_to_hour)
    return data_df


def check_no_missing_value(data_df, column, df_name, col_type):
    result = data_df[column].isnull().values.any()

    if result:
        print(f"The column {column} has missing values in {df_name}. Fill them with None/0.0")
        if col_type == 'categorical':
            data_df[column] = data_df[column].fillna('None')
        elif col_type == 'numerical':
            data_df[column] = data_df[column].fillna(0.0)
        else:
            raise Exception("Invalid col_type for check_no_missing_value!")
    else:
        print(f"The column {column} has no missing values in {df_name}")


def normalize_gaussian(data_df, column):
    value = data_df[column].to_numpy().reshape(-1, 1)
    scaler = preprocessing.StandardScaler().fit(value)
    value = scaler.transform(value)
    data_df[f'{column}_gaussian'] = value
    return data_df, scaler.mean_, scaler.scale_


def zip_lists_into_dict(list1, list2):
    codes_dict = {}
    for k, v in zip(list1, list2):
        codes_dict[k] = v

    return codes_dict, v + 1


def merge_and_divide(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    result = {}

    for k, v in dict1.items():
        result[k] = v / dict2[k]

    return result
