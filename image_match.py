# -*- coding:utf-8 -*-

import ssl
import cv2
import time
import random
import requests
import numpy as np
from urllib.request import urlopen


def get_distance(fg, bg, resize_num=1):
    fg_obj = download_imdecode(src=fg)
    fg_gray = cv2.cvtColor(fg_obj, cv2.COLOR_BGR2GRAY)
    bg_obj = download_imdecode(bg, flag=0)
    res = cv2.matchTemplate(fg_gray, bg_obj, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
    distance = max_indx[0] * resize_num
    return int(distance / 600 * 400)


def download_imdecode(src, flag=3):
    context = ssl._create_unverified_context()
    resp = urlopen(url=src, context=context)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, flag)
    return image


def get_trace(distance):
    start_time = int(time.time() * 1000)
    time.sleep(random.uniform(0.01, 0.05))
    back = random.randint(2, 6)
    distance += back
    v = 0
    tracks_list = []
    current = 0
    while current < distance - 13:
        a = random.randint(10000, 12000)  # 加速运动
        v0 = v
        t = random.randint(9, 18)
        s = v0 * t / 1000 + 0.5 * a * ((t / 1000) ** 2)
        current += s
        v = v0 + a * t / 1000
        if current < distance:
            tracks_list.append(round(current))
    if round(current) < distance:
        for i in range(round(current) + 1, distance + 1):
            tracks_list.append(i)
    else:
        for i in range(tracks_list[-1] + 1, distance + 1):
            tracks_list.append(i)
    for _ in range(back):
        current -= 1
        tracks_list.append(round(current))
    tracks_list.append(round(current) - 1)
    if tracks_list[-1] != distance - back:
        tracks_list.append(distance - back)
    timestamp_list = []
    timestamp = int(time.time() * 1000)
    for i in range(len(tracks_list)):
        t = random.randint(11, 18)
        timestamp += t
        timestamp_list.append(timestamp)
        i += 1
    y_list = []
    zy = 0
    for j in range(len(tracks_list)):
        y = random.choice(
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, -1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, -1, 0, 0])
        zy += y
        y_list.append(zy)
        j += 1
    trace = []
    for index, x in enumerate(tracks_list):
        trace.append([x, y_list[index], timestamp_list[index] - start_time])
    return trace, trace[-1][-1] + random.randint(1, 5)


class TemplateMatching(object):
    def __init__(self, bg_url, fg_url):
        for key, value in {'bg.png': bg_url, 'fg.png': fg_url}.items():
            r = requests.get(value)
            with open(key, 'wb') as f:
                f.write(r.content)
        self.bg_gray = cv2.cvtColor(cv2.imread('bg.png'), cv2.COLOR_BGR2GRAY)
        self.fg_gray = cv2.cvtColor(cv2.imread('fg.png'), cv2.COLOR_BGR2GRAY)

    @staticmethod
    def crop_fg_img(gray_img):
        gray_nparray = np.float32(gray_img)
        dst = cv2.cornerHarris(gray_nparray, 2, 3, 0.22)
        dst = cv2.dilate(dst, None)
        shape = dst.shape
        threshold = 0.01 * dst.max()
        coordinate_lst = []
        for j in range(shape[0]):
            for i in range(shape[1]):
                if dst[j][i] > threshold:
                    coordinate_lst.append(j)
        y_index_lst = []
        for i in range(len(coordinate_lst) - 1):
            if (coordinate_lst[i + 1] - coordinate_lst[i]) > 6:
                y_index_lst.append(coordinate_lst[i])
                y_index_lst.append(coordinate_lst[i + 1])
        return gray_img[y_index_lst[0] - 16:y_index_lst[-1] + 16, 2:57], (y_index_lst[0] - 18, y_index_lst[-1] + 18)

    @staticmethod
    def match(template, bg_gray):
        res = cv2.matchTemplate(bg_gray, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        left_top = max_loc
        right_bottom = (left_top[0] + 53, left_top[1] + 55)
        return int((left_top[0] + right_bottom[0]) / 2)

    @staticmethod
    def bg_img_crop(bg_gray, y_index):
        bg_gray_crop = bg_gray[y_index[0]: y_index[1], :]
        return bg_gray_crop

    def entrance(self):
        try:
            template_data = self.crop_fg_img(self.fg_gray)
            template = template_data[0]
            y_index = template_data[1]
            bg_gray_crop = self.bg_img_crop(self.bg_gray, y_index)
            location = self.match(template, bg_gray_crop)
            return int(location / 600 * 400) - 20
        except IndexError:
            return None
