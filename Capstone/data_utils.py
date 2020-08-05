import os
import re
import shutil
import csv
import glob
import pathlib
from subprocess import call
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
import matplotlib.pyplot as plt


def split_dataset(dir, train_file, test_file):
    '''
        Transform dataset from raw download to different Train and Test directories
        based on train/test split files
        :param dir: Dataset parent directory
        :param train_file: train_split file
        :param test_file: test_split file
        :return: None
    '''
    f = open(train_file, 'r')
    for x in f:
        class_dir, filename = x.split('/')
        filename, _ = filename.split()
        print(class_dir + " : " + filename)
        if not os.path.exists(os.path.join(dir, class_dir)):
            os.makedirs(os.path.join(dir, class_dir))
        shutil.move(os.path.join(dir, filename), os.path.join(os.path.join(dir, class_dir), filename))
    f.close()

    f = open(test_file, 'r')
    for x in f:
        class_dir, filename = x.split('/')
        filename = filename.rstrip()
        print(class_dir + " : " + filename)
        if not os.path.exists(os.path.join(dir, class_dir)):
            os.makedirs(os.path.join(dir, class_dir))
        shutil.move(os.path.join(dir, filename), os.path.join(os.path.join(dir, class_dir), filename))
    f.close()


def extract_frames(path, save_frame_path=None, total_frames=40, skip_frames=5):
    """
        Extract individual frames from video file. If given a directory, save frames to it
        :param path: Path to video file
        :param save_frame_path: Path to save extracted frames
        :param total_frames: Total number of frames to extract
        :return: numpy array of extracted frames
    """
    vid = cv2.VideoCapture(path)
    all_frames = 0
    count = 0
    frames = []
    vid_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if not vid.isOpened():
        print("Error opening video stream or file")
    while vid.isOpened() and (count < total_frames):
        if all_frames % skip_frames == 0:
            success, image = vid.read()
            if success:
                frames.append(image)
                if save_frame_path is not None:
                    cv2.imwrite(save_frame_path + "/frame%d.jpg" % count, image)
                count += 1
                if count > vid_length:
                    break;
            else:
                break
        all_frames += 1

    return np.concatenate(frames, axis=0)


def split_classes(anno_dir, num_classes=10):
    """
        Randomly select n random classes from list
        :param anno_dir: Class index and label annotation list file
        :param num_classes: number of random classes to select
        :return: list of selected classes
    """
    f = open(anno_dir + '/classInd.txt')
    class_list = []
    for x in f:
        cls, name = x.split()
        class_list.append(name)
        # print(name)
    f.close()

    classes = random.choices(class_list, k=num_classes)
    print(classes)
    return classes


def process_dataset(data_dir, anno_dir, save_dir):
    """
        Process dataset based on class list, extract frames, and save frames to dir
        :param data_dir: Dataset directory
        :param anno_dir: TrainTest Split and class indices directory
        :param class_list: List of randomly selected classes
        :param save_dir: Directory to save frames
        :param split: Which train/test split to select based on UCF101 Dataset
        :return: None
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Get frames for training set
    f = open(anno_dir + "/trainlist01.txt", 'r')
    for x in f:
        class_dir, filename = x.split('/')
        filename, _ = filename.split()
        vid_path = os.path.join(class_dir, filename)
        frame_path = os.path.join(class_dir, filename.split('.')[0])
        if not os.path.exists(os.path.join(save_dir, frame_path)):
            os.makedirs(os.path.join(save_dir, frame_path))

        extract_frames(os.path.join(data_dir, vid_path), os.path.join(save_dir, frame_path))

    # Get frames for test set
    f = open(anno_dir + "/testlist01.txt", 'r')
    for x in f:
        class_dir, filename = x.split('/')
        # print(x)
        vid_path = os.path.join(class_dir, filename)
        frame_path = os.path.join(class_dir, filename.split('.')[0])
        print(frame_path)
        if not os.path.exists(os.path.join(save_dir, frame_path)):
            os.makedirs(os.path.join(save_dir, frame_path))

        extract_frames(os.path.join(data_dir, vid_path), os.path.join(save_dir, frame_path))


def eda_dataset(dataset):


if __name__ == '__main__':
    print("Tensorflow Version: {}".format(tf.__version__))
    dataset_name = 'ucf101'
    print(tf.config.list_physical_devices('CPU'))
    print(tf.config.list_physical_devices('GPU'))
    dir = './UCF101'

    # split_dataset(dir, os.path.join(dir, 'trainlist01.txt'), os.path.join(dir, 'testlist01.txt'))
    eda_dataset(train)
