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

dataset = read_csv('video_scenes.csv',',')
dataset.head()

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
      # print('pred: [', str(int(pred[0])), ';', str(int(pred[1])), ']', ', true: ', test[0:2])
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

  print('Results of comparison')
  print("SSIM average for Model: ", ssim_res_1)
  print("SSIM average for [0:0]: ", ssim_res_2)
  print("SSIM average for Best:  ", ssim_res_3)
  print("Summary size for Model: ", size_res_1)
  print("Summary size for [0:0]: ", size_res_2)
  print("Summary size for Best:  ", size_res_3)


def get_dict_deblock_scenes():
  with open('ssim_for_scenes.json', 'r') as json_file:
    data = json.load(json_file)
    return data

def str_to_int(text):
  res = int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (10 ** 6)
  return res

# dataset.set_index('Scene', inplace = True)
# dataset.head()
dataset['Scene'] = dataset['Scene'].apply(lambda s:str_to_int(s))

trg = dataset[['sharpness','thrash_hold', 'Scene']]
trn = dataset.drop(['sharpness','thrash_hold', 'Scene'], axis=1)

Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.2)

def distance(x1, y1, x2, y2):
  return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

model = LinearRegression()

model.fit(Xtrn, Ytrn[['sharpness', 'thrash_hold']])
preds = model.predict(Xtest)
preds = np.round(preds)

sharp_err = []
thrash_hold_err = []
dist = []
ys = np.array(Ytest)
for pred, y in zip(preds, ys):
  sharp_err.append(np.abs(pred[0] - y[0]))
  thrash_hold_err.append(np.abs(pred[1] - y[1]))
  dist.append(distance(pred[0], pred[1], y[0], y[1]))
print(f'Result of model {model}')
print(np.mean(sharp_err))
print(np.mean(thrash_hold_err))
print(np.mean(dist))

result_analyze(get_dict_deblock_scenes(), model, np.array(Xtest), np.array(Ytest))

model = RandomForestRegressor(n_estimators=100, max_features ='sqrt')

model.fit(Xtrn, Ytrn[['sharpness', 'thrash_hold']])
preds = model.predict(Xtest)
preds = np.round(preds)
sharp_err = []
thrash_hold_err = []
dist = []
ys = np.array(Ytest)
for pred, y in zip(preds, ys):
  sharp_err.append(np.abs(pred[0] - y[0]))
  thrash_hold_err.append(np.abs(pred[1] - y[1]))
  dist.append(distance(pred[0], pred[1], y[0], y[1]))
print(f'Result of model {model}')
print('Feature importance:', model.feature_importances_)
print(np.mean(sharp_err))
print(np.mean(thrash_hold_err))
print(np.mean(dist))

result_analyze(get_dict_deblock_scenes(), model, np.array(Xtest), np.array(Ytest))

model = KNeighborsRegressor(n_neighbors=6)

model.fit(Xtrn, Ytrn[['sharpness', 'thrash_hold']])
preds = model.predict(Xtest)
preds = np.round(preds)
sharp_err = []
thrash_hold_err = []
dist = []
ys = np.array(Ytest)
for pred, y in zip(preds, ys):
  sharp_err.append(np.abs(pred[0] - y[0]))
  thrash_hold_err.append(np.abs(pred[1] - y[1]))
  dist.append(distance(pred[0], pred[1], y[0], y[1]))
print(f'Result of model {model}')
print(np.mean(sharp_err))
print(np.mean(thrash_hold_err))
print(np.mean(dist))

result_analyze(get_dict_deblock_scenes(), model, np.array(Xtest), np.array(Ytest))

from collections import Counter
import matplotlib.pyplot as plt

labels = np.array(dataset[['sharpness', 'thrash_hold']]).tolist()
d = {}
for i in [-3, -2, -1, 0, 1, 2, 3]:
  d[i] = {}
for i in [-3, -2, -1, 0, 1, 2, 3]:
  for j in [-3, -2, -1, 0, 1, 2, 3]:
    d[i][j] = 0

for elem in labels:
  d[elem[0]][elem[1]] += 1

x, y = np.mgrid[-3:3:7j, -3:3:7j]
print(x)
z = []

for i in [-3, -2, -1, 0, 1, 2, 3]:
  z.append([])
  for j in [-3, -2, -1, 0, 1, 2, 3]:
    print('[', i, ':', j, '] - ', d[i][j])
    z[i+3].append(d[i][j])
z = np.array(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='inferno')
ax.legend()