import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


SEED = 8642
np.random.seed(SEED)

config = {'min_xval':55.55, 'max_xval':55.95, 'min_yval':37.3, 'max_yval':37.9, 'x_ngroups': 8, 'y_ngroups': 8}

def split_on_intervals(min_val, max_val, n):
    step = (max_val - min_val)/n
    intervals = [min_val+(step*x) for x in range(n+1)]
    return intervals


def create_groups(x_intervals, y_intervals):
    groups = {}
    x_intervals = np.concatenate([[-np.inf], x_intervals, [np.inf]])
    y_intervals = np.concatenate([[-np.inf], y_intervals, [np.inf]])
    for x_i in range(len(x_intervals)-1):
        for y_i in range(len(y_intervals)-1):
            groups[f'x : {x_intervals[x_i]} - {x_intervals[x_i+1]} | y : {y_intervals[y_i]} - {y_intervals[y_i+1]}'] = 0
    return groups


#Сортирует точки по регионам
def sort_on_groups(x_vals, y_vals, x_intervals, y_intervals, groups, only_vals = False):
    for x, y in zip(x_vals, y_vals):
        for x_i in range(len(x_intervals)-1):
            for y_i in range(len(y_intervals)-1):
                if ((x_intervals[x_i] <= x < x_intervals[x_i+1]) and (y_intervals[y_i] <= y < y_intervals[y_i+1])):
                    groups[f'x : {x_intervals[x_i]} - {x_intervals[x_i+1]} | y : {y_intervals[y_i]} - {y_intervals[y_i+1]}'] += 1

    if only_vals:
        return list(groups.values())

    return groups


def create_dataset(config, df):
    x_intervals = split_on_intervals(config['min_xval'], config['max_xval'], config['x_ngroups'])
    y_intervals = split_on_intervals(config['min_yval'], config['max_yval'], config['y_ngroups'])
    groups = create_groups(x_intervals, y_intervals)

    groups_values = []
    for i in range(len(df)):
        g = df.iloc[i]
        points = np.array([[float(x['lat']), float(x['lon'])] for x in g['points']])
        group_values = sort_on_groups(points[:,0], points[:,1], x_intervals, y_intervals, groups.copy(), only_vals = True)
        groups_values.append(group_values)

    groups_values = np.array(groups_values)
    for i in range(len(groups.keys())):
        groups[list(groups.keys())[i]]=groups_values[:,i]

    return groups


def feature_engineering(df: pd.DataFrame, n_groups: int = 8, parse_json: bool = True, dists: bool = True, 
                        age_feats: bool = True, normalize: bool = True, drop_zero_cols: bool = False,
                        process_points: bool = True) -> pd.DataFrame:

    df.reset_index(drop=True, inplace=True)

    if parse_json:
        df_norm = pd.concat([df, pd.json_normalize(df['targetAudience'])], axis=1)
        # print(df_norm)
        df_norm = df_norm.drop(columns=['targetAudience'])
    else:
        df_norm = df

    if process_points:
        df_norm['points_count'] = df_norm['points'].apply(len)
        if dists:
            lat_center = 55.7522
            lon_center = 37.6156
            # print(df_norm)
            df_norm['distance_center'] = df_norm['points'].apply(lambda points: [((float(point['lat']) - lat_center) ** 2 + (float(point['lon']) - lon_center) ** 2) ** 0.5 for point in points]).apply(sum) / df_norm['points_count']
            lat_patr = 55.763868
            lon_patr = 37.592168
            df_norm['distance_patriki'] = df_norm['points'].apply(lambda points: [((float(point['lat']) - lat_patr) ** 2 + (float(point['lon']) - lon_patr) ** 2) ** 0.5 for point in points]).apply(sum) / df_norm['points_count']
            lat_luzh = 55.717934
            lon_luzh = 37.551932
            df_norm['distance_luzhniki'] = df_norm['points'].apply(lambda points: [((float(point['lat']) - lat_luzh) ** 2 + (float(point['lon']) - lon_luzh) ** 2) ** 0.5 for point in points]).apply(sum) / df_norm['points_count']

        config = {'min_xval': 55.55, 'max_xval': 55.95, 'min_yval': 37.3, 'max_yval': 37.9, 'x_ngroups': 8, 'y_ngroups': 8}
        dataset = pd.DataFrame(create_dataset(config, df_norm))
        if drop_zero_cols:
            zero_columns = dataset.sum()[dataset.sum() == 0].index.tolist()
            dataset = dataset.drop(columns=zero_columns)
    else:
        dataset = pd.DataFrame()

    df_new = df_norm
    for col in ['hash', 'points', 'name']:
        try:
            df_new = df_new.drop(columns=[col])
        except: pass

    if age_feats:
        df_new['age_span'] = df_new.apply(lambda row: (row['ageTo'] - row['ageFrom']) ** 1/2, axis=1)
        df_new['age_mean'] = df_new.apply(lambda row: ((row['ageTo'] + row['ageFrom']) / 2) ** 1/2, axis=1)

    gender_ohe = pd.get_dummies(df_new['gender'])
    try: 
        income_ohe = pd.get_dummies(df_new['income']).drop(columns=['ac'])
    except:
        print('no ac in income')
        income_ohe = pd.get_dummies(df_new['income'])
    
    df_full = pd.concat([dataset, df_new, gender_ohe, income_ohe], axis=1).drop(columns=['gender', 'income'])
    df_full = df_full.fillna(0)

    if normalize:
        df_scaled = (df_full - df_full.mean()) / df_full.std()
        df_scaled = df_scaled.fillna(0)
        return df_scaled
    else:
        return df_full


def fit_to_base_cols(df_scaled: pd.DataFrame, base_cols) -> pd.DataFrame:
    # base_df = pd.DataFrame(columns=base_cols)
    # for col in df_scaled.columns:
    #     base_df[col] = df_scaled[col]
    # base_df = base_df.fillna(0)
    missing_cols = list(set(base_cols.tolist()) - set(df_scaled.columns.tolist()))
    for col in missing_cols:
        df_scaled[col] = 0
    return df_scaled


def prep_test_df(test_df: pd.DataFrame, base_cols) -> pd.DataFrame:
    sc_df = feature_engineering(test_df, parse_json=True, normalize=False, drop_zero_cols=False)
    sc_df = fit_to_base_cols(sc_df, base_cols)
    return sc_df


def ensemble_inference(models, X_test):
    y_pred = np.zeros(len(X_test))
    for model in models[:]:
        y_pred += model.predict(X_test)
    y_pred /= len(models)
    return y_pred