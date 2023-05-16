import csv
import statistics

import tabulate
import statistics
import json
import numpy as np
import subprocess
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path

from feature_extractor import SITI, LapBlur
from video_parser import VideoParser
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


class info:
    def __init__(self, param: str, size: int, avg_vqmt_ssim: float, min_vqmt_ssim: float):
        self.param = param
        self.size = size
        self.avg_vqmt_ssim = avg_vqmt_ssim
        self.min_vqmt_ssim = min_vqmt_ssim


def param_to_index(param: int):  # TODO()
    d = {-3: 0,
         -2: 1,
         -1: 2,
         0: 3,
         1: 4,
         2: 5,
         3: 6}
    return d[param]


def index_to_param(index: int):  # TODO()
    d = {0: -3,
         1: -2,
         2: -1,
         3: 0,
         4: 1,
         5: 2,
         6: 3}
    return d[index]


def parse_json(json_file: str, first_arg: str, p: str, n: int, start: int, end: int):
    r1: list = [[0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n]  # change
    r2: list = [[0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n]  # change
    r3: list = [[0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n]  # change
    r4: list = [[0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n]  # change
    with open(json_file, 'r') as js_file:
        data = json.load(js_file)
    for exp in np.arange(1, 50):  # TODO()
        experiment = data[first_arg]["experiments"][exp]
        param = experiment["parameters"]
        size = sum(experiment["ladders"][p]["metrics"]["entropy"]["raw"][start:end])
        avg_vqmt_ssim = statistics.mean(experiment["ladders"][p]["metrics"]["vqmt_ssim"]["raw"][start:end])
        min_vqmt_ssim = min(experiment["ladders"][p]["metrics"]["vqmt_ssim"]["raw"][start:end])
        i = int(re.findall(r"[-+]?(?:\d*\.*\d+)", param)[-2])
        j = int(re.findall(r"[-+]?(?:\d*\.*\d+)", param)[-1])
        # print(re.findall(r"[-+]?\d*:[-+]?\d*", param)[0])
        r1[param_to_index(i)][param_to_index(j)] = re.findall(r"k [-+]?\d*:[-+]?\d*", param)[0]
        r2[param_to_index(i)][param_to_index(j)] = int(size)
        r3[param_to_index(i)][param_to_index(j)] = avg_vqmt_ssim
        r4[param_to_index(i)][param_to_index(j)] = min_vqmt_ssim
    return np.array(r1), np.array(r2), np.array(r3), np.array(r4)


def get_entropy(json_file: str, first_arg: str, p: str, start: int, end: int):
    with open(json_file, 'r') as js_file:
        data = json.load(js_file)
        experiment = data[first_arg]["experiments"][0]
        return statistics.mean(experiment["ladders"][p]["metrics"]["entropy"]["raw"][start:end])


def scene_detect(video_url: str, video_name: str, diff_percent: float):
    print('Start scene detect')
    # subprocess.run(['./ffmpeg', '-i', video_url, '-filter_complex',
    #                  f"select='gt(scene, {diff_percent})',metadata=print:file=scenes_{video_name}.log",
    #                  '-f', 'null', "-"])
    # print('End scene detect')
    return f'scenes/scenes_{video_name}.log'


def out_reader(log_file: str) -> list[int]:
    scenes = []
    with open(log_file, 'r') as log:
        lines = [line.rstrip() for line in log]
        for line in lines:
            if line[0] == 'l':
                continue
            time = float(re.findall(r"[-+]?(?:\d*\.*\d+)", line)[-1])
            scenes.append(round(time * 24))
    return scenes


def best_ssim_compression(p, a):
    maxx = a[0][0]
    ii, jj = 0, 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] > maxx:
                maxx, ii, jj = a[i][j], i, j
    return index_to_param(ii), index_to_param(jj)


def csv_format(writer, p, s, a, m):
    writer.writerows([best_ssim_compression(p, s, a)])
    writer.writerows(['-------'])
    writer.writerows(p)
    writer.writerows(['-------'])
    writer.writerows(s)
    writer.writerows(['-------'])
    writer.writerows(a)
    writer.writerows(['-------'])
    writer.writerows(m)
    writer.writerows(['-------'])
    writer.writerows(['-------'])
    writer.writerows(['-------'])
    writer.writerows(['-------'])


def csv_format_compare(writer, p, s1, a1, m1, s2, a2, m2):
    i1, j1 = best_ssim_compression(p, s1, a1)
    i2, j2 = best_ssim_compression(p, s2, a2)
    writer.writerows([str(i1 - i2), str(j1 - j2)])
    writer.writerows(['-------'])
    writer.writerows(p)
    writer.writerows(['-------'])
    writer.writerows(s1 - s2)
    writer.writerows(['-------'])
    writer.writerows(a1 - a2)
    writer.writerows(['-------'])
    writer.writerows(m1 - m2)
    writer.writerows(['-------'])
    writer.writerows(['-------'])
    writer.writerows(['-------'])
    writer.writerows(['-------'])


def extract_features(video: str):
    vp = VideoParser(video)
    vp.analyze()


def test_writer():
    video = 'convertedkln90p4t.mp4'  # 'tearsofsteel_4k.mp4'

    # scene_detect(video, 0.35)
    # extract_features(video)
    scenes = out_reader('scenes/scene_change_disney.log')
    with open('result/test.csv', 'w') as fp:
        with open('result/test_diff.csv', 'w') as sp:
            writer = csv.writer(fp)
            writer_diff = csv.writer(sp)
            for i in range(len(scenes) - 1):
                # p1, s1, a1, m1 = parse_json('deblock_ToS_[-3;3]_1920x1080.json', "s3://elty/tearsofsteel_4k.mp4", "1080p", 7,
                #                         scenes[i], scenes[i + 1])
                # p2, s2, a2, m2 = parse_json('deblock_ToS_[-3;3]_1280x720.json', "s3://elty/tearsofsteel_4k.mp4", "720p", 7,
                #                         scenes[i], scenes[i + 1])
                p1, s1, a1, m1 = parse_json('deblock_info/deblock_disney_[-3;3]_1920x1280.json',
                                            "https://ott-sources.s3.mds.yandex.net/Disney/TestReelFiles/DisneyTestClip_SDR_ProRes422HQ.mov",
                                            "1080p", 7,
                                            scenes[i], scenes[i + 1])
                p2, s2, a2, m2 = parse_json('deblock_info/deblock_disney_[-3;3]_1080x720.json',
                                            "https://ott-sources.s3.mds.yandex.net/Disney/TestReelFiles/DisneyTestClip_SDR_ProRes422HQ.mov",
                                            "720p",
                                            7,
                                            scenes[i], scenes[i + 1])
                csv_format(writer, p1, s1, a1, m1)
                csv_format_compare(writer_diff, p1, s1,
                                   np.array(a1), np.array(m1),
                                   s2,
                                   np.array(a2), np.array(m2))
                print(i, '/ {}'.format(len(scenes)))
    # vizualization()


if __name__ == '__main__':
    # test_writer()

    # videos = ["tearsofsteel_4k.mp4", "disney_1280x720.mp4"]
    # jjsons = ["deblock_info/deblock_ToS_[-3;3]_1920x1080.json", "deblock_info/deblock_disney_[-3;3]_1920x1280.json"]
    # urls = ["s3://elty/tearsofsteel_4k.mp4", "https://ott-sources.s3.mds.yandex.net/Disney/TestReelFiles/DisneyTestClip_SDR_ProRes422HQ.mov"]
    # ladders = ["1080p", "1080p"]

    # videos = ["tearsofsteel_4k.mp4"]
    # jjsons = ["deblock_info/deblock_ToS_[-3;3]_1920x1080.json"]
    # urls = ["s3://elty/tearsofsteel_4k.mp4"]
    # ladders = ["1080p"]

    # videos = ['AoT1920x1080.mov']
    # jjsons = ['deblock_info/deblock_Aot.json']
    # urls = ['https://s3.mds.yandex.net/ott/DentzuEntertainment/ataka_titanov_coid749374/s04/ataka_titanov_coid749374_s04e28_sdr_r1920x1080p23_arus2rus2rus2jpn2_d2204051120.mov']
    # ladders = ["1080p"]

    videos = ['Doctor_Haus.mp4']
    jjsons = ['deblock_info/deblock_Doctor_Haus.json']
    urls = ['https://s3.mds.yandex.net/ott/Universal/House_rus_eng_coid178710/s08/Doktor_Haus_coid178710_s08e22.mp4']
    ladders = ["1080p"]

    # videos = ['Supernatural.mp4']
    # jjsons = ['deblock_info/deblock_Supernatural.json']
    # urls = ['https://s3.mds.yandex.net/ott/Warner/Supernatural_coid178707/s15/sverhestestvennoe_coid178707_s15e21_sdr_r1920x1080p23_arus2rus6rus6eng2eng6_d2103051402.mov']
    # ladders = ["1080p"]

    print(len(videos), len(jjsons), len(urls), len(ladders))
    with open("result/video_scenes.csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(
            ["Scene", "entropy", "si", "ti", "lap_var", "lap_var_var", "lap_p25", "lap_p50", "lap_p75", "sharpness",
             "thrash_hold"])
        for video, jjson, url, ladder in zip(videos, jjsons, urls, ladders):
            print('Start video ', video)
            scene_path = scene_detect('videos/' + video, video, 0.35)
            scenes = out_reader(scene_path)
            siti = SITI()
            lap_blur = LapBlur()

            print("Start SI, TI counting")
            siti.analyze("videos/" + video, video)
            print("End SI, TI counting")

            print('Start LapBlur counting')
            lap_blur.analyze("videos/" + video, video)
            print('End LapBlur counting')

            for i in range(len(scenes) - 1):
                if i % 5 == 0:
                    print(i, 'scenes done')
                p, s, a, m = parse_json(jjson, url, ladder, 7, scenes[i], scenes[i + 1])
                entropy = get_entropy(jjson, url, ladder, scenes[i], scenes[i + 1])
                sh, thr = best_ssim_compression(p, a)
                si, ti = siti.get_info_slice(scenes[i], scenes[i + 1])
                lap_var, lap_var_var, lap_p25, lap_p50, lap_p75 = lap_blur.get_info_slice(scenes[i], scenes[i + 1])
                file_writer.writerow(
                    [f'{video}_{i}', entropy, si, ti, lap_var, lap_var_var, lap_p25, lap_p50, lap_p75, sh, thr])

    # scene_detect(video, 0.35)
