import csv

from feature_extractor import SITI


class VideoParser:
    def __init__(self, video_path):
        self.video_path: str = video_path

    def analyze(self):
        with open("video_features.csv", "w+") as data_file:
            writer = csv.writer(data_file)
            si_result = []
            ti_result = []
            si = SITI()
            si.analyze(self.video_path)
            si, ti = si.get_info()
            print(si)
            print(ti)
            writer.writerows([self.video_path, si, ti])




