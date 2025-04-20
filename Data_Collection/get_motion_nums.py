#%%
import joblib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import pandas as pd

import csv
def read_third_column(file_path):
    third_column = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >- 3:
                third_column.append(row[2])
    return third_column

def read_frame_num(file_path):
    start_frame = []
    end_frame = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            start_frame.append(row[0])
            end_frame.append(row[1])
    return start_frame, end_frame

motion_name_path = "./Data_Collection/mujoco/frame_text.csv"
motion_names = read_third_column(motion_name_path)
start_frames, end_frames = read_frame_num(motion_name_path)
# print(motion_names)
new_motion_names = []
new_start_frames = []
new_end_frames = []
# print(motion_names)


class_list = [
['wave'],
['pour and mix', 'pour', 'mix', 'whisk'],

['reach out', 'stretch', 'point', 'hold hands', 'straighten', 'put objects','put the cup','put hands','put the cup', 'place'],
['stir'],
['violin'],
['wipe'],
['guitar'],
['drink'],
['shower'],

['hand through hair', 'hand to mouth'],
['cut', 'sponge', 'drum'],
['wash'],
['draw circle', 'do TaiChi'],
['dry'],
['tennis'],
['open'],

['take of t-shirt'],
['take book from shelf'],
['clap', 'slap'],

['golf'],
['punch'],
['make a bow', 'cross'],
['knife', 'cup'],
['throw'],


['shake'],

['cast'],

['high five'],
['make the stop sign'],
['chop'],
['salute'],
    
]

class_new_number_list = [
    1,1,
    2,2,2,2,2,2,2,
    3,3,3,3,3,3,3,
    4,4,4,
    5,5,5,5,5,
    6,
    10,10,10,
    20,20
]

class_idx_list = []
class_frame_num_list = []
for i in range(len(class_list)):
    class_frame_num_list.append(0)

import joblib
motion_file_names = joblib.load("./Data_Collection/select_data/select_motion_name.pkl")


motion_nums_dict = {}
nnn = 0
no = 0
yes = 0
# 对于每一个motion，保存其需要重复的次数
for i in range(len(motion_names)):
    


    flag = -1
    for j in range(len(class_list)):
        for k in range(len(class_list[j])):
            if class_list[j][k] in motion_names[i]:
                class_idx_list.append(j)
                class_frame_num_list[j] += int(end_frames[i]) - int(start_frames[i])
                flag = j
                break
        if flag != -1:
            break
    if flag == -1: #说明是no
        class_idx_list.append(-1)
    numss = 1
    if class_idx_list[i] != -1:
        numss = class_new_number_list[class_idx_list[i]]
    else:
        numss = 0
    motion_nums_dict.update({motion_file_names[i]: numss})

# print(motion_nums_dict)
joblib.dump(motion_nums_dict, "./Data_collection/motion_num_dict.pkl")