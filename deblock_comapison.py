import argparse
import json
import re
import statistics
import hashlib

import numpy as np

from main import scene_detect, out_reader, parse_json

jjsons = ["deblock_info/deblock_Doctor_Haus.json",
          "deblock_info/deblock_ToS_[-3;3]_1920x1080.json",
          "deblock_info/deblock_disney_[-3;3]_1920x1280.json",
          "deblock_info/deblock_Aot.json",
          "deblock_info/deblock_Supernatural.json"]

ladders = ["1080p", "1080p", "1080p", "1080p", "1080p"]

urls = ["https://s3.mds.yandex.net/ott/Universal/House_rus_eng_coid178710/s08/Doktor_Haus_coid178710_s08e22.mp4",
        "s3://elty/tearsofsteel_4k.mp4",
        "https://ott-sources.s3.mds.yandex.net/Disney/TestReelFiles/DisneyTestClip_SDR_ProRes422HQ.mov",
        "https://s3.mds.yandex.net/ott/DentzuEntertainment/ataka_titanov_coid749374/s04/ataka_titanov_coid749374_s04e28_sdr_r1920x1080p23_arus2rus2rus2jpn2_d2204051120.mov",
        "https://s3.mds.yandex.net/ott/Warner/Supernatural_coid178707/s15/sverhestestvennoe_coid178707_s15e21_sdr_r1920x1080p23_arus2rus6rus6eng2eng6_d2103051402.mov"]

deb = {}
for i in [-3, -2, -1, 0, 1, 2, 3]:
    deb[i] = {}
for i in [-3, -2, -1, 0, 1, 2, 3]:
    for j in [-3, -2, -1, 0, 1, 2, 3]:
        deb[i][j] = {}
        deb[i][j]['ssim'] = 0
        deb[i][j]['size'] = 0
        deb[i][j]['ssim_var'] = 0

x_ssim = 0
x_size = 0
x_ssim_var = 0

for jjson, ladder, url in zip(jjsons, ladders, urls):
    print(f'Start {jjson}')
    with open(jjson, 'r') as js_file:
        deblock_data = json.load(js_file)

    exp = 0
    experiment = deblock_data[url]["experiments"][exp]
    param = experiment["parameters"]
    size = sum(experiment["ladders"][ladder]["metrics"]["entropy"]["raw"])
    avg_vqmt_ssim = statistics.mean(experiment["ladders"][ladder]["metrics"]["vqmt_ssim"]["raw"])
    var_vqmt_ssim = statistics.variance(experiment["ladders"][ladder]["metrics"]["vqmt_ssim"]["raw"])
    # sharp = int(re.findall(r"[-+]?(?:\d*\.*\d+)", param)[-2])
    # thrash = int(re.findall(r"[-+]?(?:\d*\.*\d+)", param)[-1])

    x_ssim += avg_vqmt_ssim
    x_size += size
    x_ssim_var += var_vqmt_ssim

x = len(jjsons)

x_ssim /= x
x_size /= x
x_ssim_var /= x
print(f'For [0:0]:')
print(f'ssim: {x_ssim}')
print(f'size: {x_size}')
print(f'ssim_var: {x_ssim_var}')

# for i in [-3, -2, -1, 0, 1, 2, 3]:
#     for j in [-3, -2, -1, 0, 1, 2, 3]:
#         deb[i][j]['ssim'] /= x
#         deb[i][j]['size'] /= x
#         deb[i][j]['ssim_var'] /= x
#         print('\n')
#         print(f'For [{i}:{j}]:')
#         print(f'ssim: {deb[i][j]["ssim"]}')
#         print(f'size: {deb[i][j]["size"]}')
#         print(f'ssim_var: {deb[i][j]["ssim_var"]}')
