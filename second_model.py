from pandas import read_csv, DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import json
import hashlib

from catboost import CatBoostClassifier, MultiTargetCustomObjective, MultiTargetCustomMetric, CatBoostRegressor

dataset = read_csv('video_scenes.csv',',')
dataset.head()

# def result_analyze(scene_id_dict, model, X_test, y_test):
#   y_pred = model.predict(X_test)
#   with open('model_compare.csv', 'w') as fp:
#     writer = csv.writer(fp)
#     for pred, test in zip(y_pred, y_test):

def str_to_int(text):
  res = int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (10 ** 6)
  return res

dataset['Scene'] = dataset['Scene'].apply(lambda s:str_to_int(s))

trg = dataset[['sharpness', 'thrash_hold', 'Scene']]
trn = dataset.drop(['sharpness','thrash_hold', 'Scene'], axis=1)

trg.head()

trn.head()

Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.2)


def ssim_diff(sh_1, th_1, scene_1, sh_2, th_2, scene_2) -> float:
  with open('ssim_for_scenes.json', 'r') as json_file:
    data = json.load(json_file)
    print('scene_1: ', scene_1)
    print('sh_1: ', sh_1)
    print('sh_2: ', sh_2)
    pred = data[str(int(scene_1))][str(int(sh_1))][str(int(th_1))]['ssim']
    true = data[str(int(scene_2))][str(int(sh_2))][str(int(th_2))]['ssim']
    return true - pred

class MultiRmseObjective(MultiTargetCustomObjective):
    def calc_ders_multi(self, approx, target, weight):
        assert len(target) == len(approx)

        w = weight if weight is not None else 1.0
        # der1 = [(target[i] - approx[i]) * w for i in range(len(approx))]
        der1 = [(target[0] - approx[0]) * w, (target[0] - approx[0]) * w, 0]
        # der2 = [-w for i in range(len(approx))]
        der2 = [-w, -w, 0]

        return (der1, der2)


class MultiRmseMetric(MultiTargetCustomMetric):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(target) == len(approxes)
        assert len(target[0]) == len(approxes[0])
        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approxes[0])):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            # for d in range(len(approxes)):
            #     error_sum += w * ((approxes[d][i] - target[d][i])**2)
            scene = target[2][i]
            sh_1 = approxes[0][i]
            sh_2 = target[0][i]
            th_1 = approxes[1][i]
            th_2 = target[1][i]
            with open('ssim_for_scenes.json', 'r') as json_file:
                data = json.load(json_file)
                pred = data[str(int(scene))][str(int(sh_1))][str(int(th_1))]['ssim']
                true = data[str(int(scene))][str(int(sh_2))][str(int(th_2))]['ssim']
            ssim_diff = true - pred
            error_sum += w * ssim_diff
        return error_sum, weight_sum

model2 = CatBoostRegressor(iterations=200, loss_function=MultiRmseObjective(), eval_metric=MultiRmseMetric(),
                           learning_rate=0.03, bootstrap_type='Bayesian', boost_from_average=False,
                           leaf_estimation_iterations=1, leaf_estimation_method='Gradient')

# with open('ssim_for_scenes.json', 'r') as json_file:
#   data = json.load(json_file)
#   elems = np.array(Ytrn['Scene'])
#   for elem in elems:
#     data[str(elem)]
model2.fit(Xtrn, Ytrn, eval_set=(Xtest, Ytest))


def result_analyze(scene_deblock_info, model, X_test, y_test):
  y_pred = model.predict(X_test)
  ssim_res_1 = 0
  size_res_1 = 0
  ssim_res_2 = 0
  size_res_2 = 0
  ssim_res_3 = 0
  size_res_3 = 0
  with open('model_compare.csv', 'w') as fp:
    writer = csv.writer(fp)
    for pred, test in zip(y_pred, y_test):
      pred = np.round(pred)
      ssim_pred = scene_deblock_info[str(int(test[2]))][str(int(pred[0]))][str(int(pred[1]))]['ssim']
      size_pred = scene_deblock_info[str(int(test[2]))][str(int(pred[0]))][str(int(pred[1]))]['size']

      ssim_base = scene_deblock_info[str(int(test[2]))]['0']['0']['ssim']
      size_base = scene_deblock_info[str(int(test[2]))]['0']['0']['size']

      ssim_true = scene_deblock_info[str(int(test[2]))][str(int(test[0]))][str(int(test[1]))]['ssim']
      size_true = scene_deblock_info[str(int(test[2]))][str(int(test[0]))][str(int(test[1]))]['size']

      ssim_res_1 += ssim_pred
      ssim_res_2 += ssim_base
      ssim_res_3 += ssim_true
      size_res_1 += size_pred
      size_res_2 += size_base
      size_res_3 += size_true

  ssim_res_1 /= len(y_pred)
  ssim_res_2 /= len(y_pred)
  ssim_res_3 /= len(y_pred)
  print("SSIM average for Model: ", ssim_res_1)
  print("SSIM average for Basic: ", ssim_res_2)
  print("SSIM average for Label: ", ssim_res_3)
  print("Summary size for Model: ", size_res_1)
  print("Summary size for Basic: ", size_res_2)
  print("Summary size for Label: ", size_res_3)


def get_dict_deblock_scenes():
  with open('ssim_for_scenes.json', 'r') as json_file:
    data = json.load(json_file)
    return data

result_analyze(get_dict_deblock_scenes(), model2, np.array(Xtest), np.array(Ytest))

# model_1 = CatBoostClassifier(loss_function='MultiLogloss')
# model_1.fit(Xtrn,
#           Ytrn,
#           verbose=True)
# print(model_1.predict(Xtest))

# model3 = CatBoostRegressor(iterations=100, loss_function=MultiTargetCustomObjective(), eval_metric=MultiTargetCustomMetric(),
#                            learning_rate=0.03, bootstrap_type='Bayesian', boost_from_average=False,
#                            leaf_estimation_iterations=1, leaf_estimation_method='Gradient')

# # with open('ssim_for_scenes.json', 'r') as json_file:
# #   data = json.load(json_file)
# #   elems = np.array(Ytrn['Scene'])
# #   for elem in elems:
# #     data[str(elem)]
# model2.fit(Xtrn, Ytrn, eval_set=(Xtest, Ytest))