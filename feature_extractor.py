import json
import subprocess
from abc import ABC, abstractmethod
from statistics import mean
from imutils import paths
import argparse
import cv2
import numpy as np
import os, shutil

class FeatureExtractor(ABC):
    @abstractmethod
    def analyze(self, video_path, video_name):
        pass

    @abstractmethod
    def get_info(self):
        pass


class SITI(FeatureExtractor):
    def __init__(self):
        self.si = None
        self.ti = None

    def analyze(self, video_path, video_name):
        # if not os.path.exists(f'siti/siti_result_{video_path}.json'):
        # with open(f'siti/siti_result_{video_name}.json', "w") as output_file:
        #     subprocess.call(['./siti-tools', video_path, '-r',
        #                      'full'], stdout=output_file)

        with open(f'siti/siti_result_{video_name}.json', 'r') as output_file:
            data = json.load(output_file)
            si_array = data['si']
            ti_array = data['ti']
            self.si = si_array
            self.ti = ti_array

    def get_info(self):
        return np.average(self.si), np.average(self.ti)

    def get_info_slice(self, start, end):
        return np.average(self.si[start:end]), np.average(self.ti[start:end])


class LapBlur(FeatureExtractor):
    def __init__(self):
        self.blur = None
        self.p25 = None
        self.p50 = None
        self.p75 = None

    def variance_of_laplacian(self, one_image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        lapl = cv2.Laplacian(one_image, cv2.CV_64F)
        return lapl.var(), np.percentile(lapl, 25), np.percentile(lapl, 50), np.percentile(lapl, 75)

    def lap_blur(self, images_path):
        # loop over the input images
        vm, p_25m, p_50m, p_75m = [], [], [], []
        it = 1
        for imagePath in paths.list_images(images_path):
            if it % 1000 == 0:
                print('Blur iterations: ', it)
            # load the image, convert it to grayscale, and compute the
            # focus measure of the image using the Variance of Laplacian
            # method
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            v, p_25, p_50, p_75 = self.variance_of_laplacian(gray)
            vm.append(v)
            p_25m.append(p_25)
            p_50m.append(p_50)
            p_75m.append(p_75)
            it = it + 1


        return vm, p_25m, p_50m, p_75m

        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        # show the image
        # cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # cv2.imshow("Image", image)

    def clear_dir(self, dir_path):
        folder = dir_path
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def analyze(self, video_path, video_name):
        print(video_path)
        # self.clear_dir('frames')
        print('Start framing for Blur')

        # subprocess.call(['./ffmpeg', '-i', video_path, 'frames/frame%05d.bmp'])
        print('Start calculate Blur')
        self.blur, self.p25, self.p50, self.p75 = self.lap_blur('frames')
        # self.clear_dir('frames')

    def get_info(self):
        return np.average(self.blur), \
               np.var(self.blur), \
               np.average(self.p25), \
               np.average(self.p50), \
               np.average(self.p75)

    def get_info_slice(self, start, end):
        return np.average(self.blur[start:end]), \
               np.var(self.blur[start:end]), \
               np.average(self.p25[start:end]), \
               np.average(self.p50[start:end]), \
               np.average(self.p75[start:end])
