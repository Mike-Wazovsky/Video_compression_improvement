import argparse
import json
import re
import statistics
import hashlib

import numpy as np

from main import scene_detect, out_reader, parse_json

data = {}

# videos = ["disney_1280x720.mp4"]
# jjsons = ["deblock_info/deblock_disney_[-3;3]_1920x1280.json"]
# urls = ["https://ott-sources.s3.mds.yandex.net/Disney/TestReelFiles/DisneyTestClip_SDR_ProRes422HQ.mov"]
# ladders = ["1080p"]

# videos = ['AoT1920x1080.mov']
# jjsons = ['deblock_info/deblock_Aot.json']
# urls = ['https://s3.mds.yandex.net/ott/DentzuEntertainment/ataka_titanov_coid749374/s04/ataka_titanov_coid749374_s04e28_sdr_r1920x1080p23_arus2rus2rus2jpn2_d2204051120.mov']
# ladders = ["1080p"]

videos = ["Doctor_Haus.mp4", "tearsofsteel_4k.mp4", "disney_1280x720.mp4", "AoT1920x1080.mov", "Supernatural.mp4"]
jjsons = ["deblock_info/deblock_Doctor_Haus.json",
          "deblock_info/deblock_ToS_[-3;3]_1920x1080.json",
          "deblock_info/deblock_disney_[-3;3]_1920x1280.json",
          "deblock_info/deblock_Aot.json",
          "deblock_info/deblock_Supernatural.json"]
urls = ["https://s3.mds.yandex.net/ott/Universal/House_rus_eng_coid178710/s08/Doktor_Haus_coid178710_s08e22.mp4",
        "s3://elty/tearsofsteel_4k.mp4",
        "https://ott-sources.s3.mds.yandex.net/Disney/TestReelFiles/DisneyTestClip_SDR_ProRes422HQ.mov",
        "https://s3.mds.yandex.net/ott/DentzuEntertainment/ataka_titanov_coid749374/s04/ataka_titanov_coid749374_s04e28_sdr_r1920x1080p23_arus2rus2rus2jpn2_d2204051120.mov",
        "https://s3.mds.yandex.net/ott/Warner/Supernatural_coid178707/s15/sverhestestvennoe_coid178707_s15e21_sdr_r1920x1080p23_arus2rus6rus6eng2eng6_d2103051402.mov"]
ladders = ["1080p", "1080p", "1080p", "1080p", "1080p"]


def str_to_int(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 6)


hashes = []
for video, jjson, url, ladder in zip(videos, jjsons, urls, ladders):
    print('Video: ', video)
    scene_path = scene_detect("videos/" + video, video, 0.35)
    scenes = out_reader(scene_path)
    for i in range(len(scenes) - 1):
        if i % 5 == 0:
            print(i, 'scenes are done')
        data[str_to_int(f'{video}_{i}')] = {}
        hashes.append(str_to_int(f'{video}_{i}'))

        start = scenes[i]
        end = scenes[i + 1]
        with open(jjson, 'r') as js_file:
            deblock_data = json.load(js_file)
        for exp in np.arange(1, 50):
            experiment = deblock_data[url]["experiments"][exp]
            param = experiment["parameters"]
            sharp = int(re.findall(r"[-+]?(?:\d*\.*\d+)", param)[-2])

            data[str_to_int(f'{video}_{i}')][sharp] = {}

        for exp in np.arange(1, 50):
            experiment = deblock_data[url]["experiments"][exp]
            param = experiment["parameters"]
            size = sum(experiment["ladders"][ladder]["metrics"]["entropy"]["raw"][start:end])
            avg_vqmt_ssim = statistics.mean(experiment["ladders"][ladder]["metrics"]["vqmt_ssim"]["raw"][start:end])
            sharp = int(re.findall(r"[-+]?(?:\d*\.*\d+)", param)[-2])
            thrash = int(re.findall(r"[-+]?(?:\d*\.*\d+)", param)[-1])

            data[str_to_int(f'{video}_{i}')][sharp][thrash] = {}
            data[str_to_int(f'{video}_{i}')][sharp][thrash]['ssim'] = avg_vqmt_ssim
            data[str_to_int(f'{video}_{i}')][sharp][thrash]['size'] = size

with open("result/ssim_for_scenes.json", mode="w") as w_file:
    json.dump(data, w_file)
print(hashes)
