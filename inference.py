import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils import *


SEED = 8642
np.random.seed(SEED)

config = {'min_xval':55.55, 'max_xval':55.95, 'min_yval':37.3, 'max_yval':37.9, 'x_ngroups': 8, 'y_ngroups': 8}

df = pd.read_json('./train_data.json')
target = df.pop('value')

points = []
for i in df['points']:
    for j in i:
        points += [(float(j['lat']), float(j['lon']))]

points_set = set(points)

def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

def find_nearest_points(target_coord, points, n):
    distances = [(calculate_distance(target_coord, point), point) for point in points]
    distances.sort()
    return [point for dist, point in distances[:n]]

def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull(points):
    # алгоритм Джарвиса
    points = sorted(points)
    if len(points) <= 2:
        return points
    lower = []
    upper = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def process_income(income_categs: list) -> str:
    if income_categs == ['a'] or income_categs == ['b']:
        return 'ab'
    elif income_categs == ['a', 'c']:
        return 'abc'
    else:
        return ''.join(sorted(income_categs))


def sector_inference(cb_model: CatBoostClassifier, income: list, gender: str, 
                     ageFrom: int, ageTo: int, value: float): 
    """
      income: NON-EMPTY list containing 'a', 'b', or 'c'
      gender: one of ['male', 'female', 'all']
      ageFrom: lower bound for target audience
      ageTo: upper bound
      value: campaign results
    """
    needed_cols =  ['value',
                    'ageFrom',
                    'ageTo',
                    'age_span',
                    'age_mean',
                    'all',
                    'female',
                    'male',
                    'ab',
                    'abc',
                    'bc',
                    'c']
    str_income = process_income(income)
    if str_income not in needed_cols:
        str_income = 'ab'
    if gender not in needed_cols:
        gender = 'all'

    x_intervals = split_on_intervals(config['min_xval'], config['max_xval'], config['x_ngroups'])
    y_intervals = split_on_intervals(config['min_yval'], config['max_yval'], config['y_ngroups'])

    my_df = dict()
    my_df['value'] = [value]
    my_df['ageFrom'] = [ageFrom]
    my_df['ageTo'] = [ageTo]
    my_df['age_span'] = [(np.array(my_df['ageTo']) - np.array(my_df['ageFrom'])) ** 1/2]
    my_df['age_mean'] = [((np.array(my_df['ageTo']) + np.array(my_df['ageFrom'])) / 2) ** 1/2]
    my_df['all'] = [gender == 'all']
    my_df['male'] = [gender == 'male']
    my_df['female'] = [gender == 'female']
    my_df['ab'] = [str_income == 'ab']
    my_df['abc'] = [str_income == 'abc']
    my_df['bc'] = [str_income == 'bc']
    my_df['c'] = [str_income == 'c']

    test_df = pd.DataFrame.from_dict(my_df)
    sec_id = cb_model.predict(test_df).flatten()[0]
    x_coord = x_intervals[sec_id % (config['y_ngroups'] + 2)]
    y_coord = y_intervals[sec_id // (config['x_ngroups'] + 2)] 
    return (x_coord, y_coord)


def get_point_set(df: pd.DataFrame):
    points = []
    for i in df['points']:
        for j in i:
            points += [(float(j['lat']), float(j['lon']))]
    points_set = set(points)
    return points_set


def run(model_path: str, n_neighbors: int, source_df: pd.DataFrame, income: list, 
        gender: str, ageFrom: int, ageTo: int, value: float):
    cb_model = CatBoostClassifier()
    cb_model.load_model(model_path)

    target_coord = sector_inference(cb_model, income=income, gender=gender, 
                                    ageFrom=ageFrom, ageTo=ageTo, value=value)  ####
    points_set = get_point_set(source_df)
    nearest_points = find_nearest_points(target_coord, points_set, n_neighbors)
    return nearest_points


if __name__ == '__main__':
    res = run(model_path='./catboost_for_polygons', 
              n_neighbors=11,
              source_df=df,
              income=['a', 'b'],
              gender='all',
              ageFrom=50,
              ageTo=100,
              value=77.7)
    print(res)