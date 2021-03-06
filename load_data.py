import numpy as np
import copy
import os

ROOT_DIR = '/home/ych/Downloads/UCRArchive_2018/'
TS_NAME = 'Yoga'

def read_data():
    root_dir, ts_name = ROOT_DIR, TS_NAME
    ts_dir = os.path.join(root_dir, ts_name)

    data_train = np.loadtxt(os.path.join(ts_dir, ts_name + '_TRAIN.tsv'), delimiter='\t')
    data_test = np.loadtxt(os.path.join(ts_dir, ts_name + '_TEST.tsv'), delimiter='\t')

    label_train = np.array(data_train[:, 0], np.float32)
    label_test = np.array(data_test[:, 0], np.float32)

    input_train = np.array(data_train[:, 1:], np.float32)
    input_test = np.array(data_test[:, 1:], np.float32)

    return input_train, label_train, input_test, label_test

def redefine_category(label):
    category2real = {}
    real2category = {}
    label = np.sort(np.unique(label))
    class_num = len(label)
    for k in range(class_num):
        category2real[k] = label[k]
        real2category[label[k]] = k
    return category2real, real2category

def transform_category(label, real2category):
    transformed_label = copy.deepcopy(label)
    for k in range(len(label)):
        transformed_label[k] = real2category[label[k]]
    return transformed_label

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()
    category2real, real2category = redefine_category(y_train)
    transformed_label = transform_category(y_train, real2category)

    # print(np.any(np.isnan(x_train)))
    # print(np.any(np.isnan(y_train)))
    # print(np.any(np.isnan(x_test)))
    # print(np.any(np.isnan(y_test)))