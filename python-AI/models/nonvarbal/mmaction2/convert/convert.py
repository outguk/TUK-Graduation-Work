import argparse
import os
import json
import numpy as np
import pickle
import random

import copy
from math import cos, sin, radians
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train a action recognizer')
    parser.add_argument(
        '--output-dir',
        default='test.pkl',
        help='pkl output directory path')
    parser.add_argument(
        '--text',
        action='store_true',
        help='whether to output text file')
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Data augmentation True or False')
    parser.add_argument(
        '--aug_frame',
        action='store_true',
        help='Data augmentation True or False')

    args = parser.parse_args()
    return args

def convert_dataset(args):
    '''
    Convert our dataset to PYSKL dataset
    '''
    json_path = 'json/'

    result_json = {}
    result_json['split'] = {}
    result_json['split']['xsub_train'] = []
    result_json['split']['xsub_val'] = []
    result_json['annotations'] = []

    train_ratio = 0.8

    file_list = os.listdir(json_path)
    file_count = len(file_list)

    files = []

    agmentation_order = 5

    for i in tqdm(range(file_count)):
    # for i in tqdm(range(100)):
        # Make split field
        # Split: The value of the split field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
        frame_dir = file_list[i].replace(".json", "")

        # Record data distribution
        label = str(frame_dir.split("_")[-2].replace("B",""))


        files.append(frame_dir)
        if args.augment == True:
            for j in range(agmentation_order):
                files.append(f"{frame_dir}--{str(j)}")

        # Make annotations field
        # Annotations: The value of the annotations field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
        #   frame_dir (str): The identifier of the corresponding video.
        #   total_frames (int): The number of frames in this video.
        #   img_shape (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
        #   original_shape (tuple[int]): Same as img_shape.
        #   label (int): The action label.
        #   keypoint (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as total_frames); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
        #   keypoint_score (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.

        annotation = {}
        with open(os.path.join(json_path, file_list[i]), 'r') as f:
            try:
                json_file = json.load(f)
            except:
                print(f"[INFO] Exception in load json file : {f}")
                continue

            label = int(frame_dir.split("_")[-2].replace("B",""))

            label -= 1

            annotation['frame_dir'] = frame_dir
            annotation['label'] = label
            annotation['img_shape'] = tuple(map(int, json_file['dataset']['resolution'].split("x")[::-1]))
            annotation['original_shape'] = tuple(map(int, json_file['dataset']['resolution'].split("x")[::-1]))
            # annotation['total_frames'] = 10

            # get keypoints
            keypoints = []
            for j in json_file['annotation']:
                keypoints.append(convert_keypoint(np.array(j['keypoints'])[:,:2].tolist()))

            annotation['keypoint'] = np.array([keypoints])
            annotation['total_frames'] = np.array(keypoints).shape[0]

            if args.aug_frame == True:
                keypoint = copy.deepcopy(annotation['keypoint'][0])
                annotation['keypoint'] = np.array([augment_frame(keypoint)])
                annotation['total_frames'] = annotation['keypoint'].shape[1]

            keypoint_score = np.random.randint(7500, 9999, np.array(annotation['keypoint'].shape[:-1]))
            annotation['keypoint_score'] = keypoint_score / 10000

            result_json['annotations'].append(annotation)

            if args.augment == True:
                for z in range(agmentation_order):
                    agmentation = copy.deepcopy(annotation)
                    agmentation['frame_dir'] = f"{frame_dir}--{str(z)}"
                    rotate_data = []
                    range_rotate = random.randint(20, 160)
                    for idx in agmentation['keypoint'][0]:
                        calc = list(map(lambda x: calc_rotate(960, 540, x[0], x[1], range_rotate), idx))
                        rotate_data.append(calc)
                    agmentation['keypoint'] = np.array([rotate_data])
                    result_json['annotations'].append(agmentation)

    file_count = len(files)
    result_json['split']['xsub_train'] = copy.deepcopy(files[:int(file_count * train_ratio)])
    result_json['split']['xsub_val'] = copy.deepcopy(files[int(file_count * train_ratio):])

    if args.text == True:
        print(f"Save text data.")
        with open('test.txt', 'w') as f:
            f.write(str(result_json))
        print(f"Done.")


    with open(args.output_dir, 'wb') as f:
        print(f"Save pickle data.")
        pickle.dump(result_json, f)
        print(f"Done.")

    train_data = {}
    valid_data = {}
    for i in result_json['split']['xsub_train']:
        label = str(i.split("_")[-2].replace("B",""))
        if train_data.get(label) == None: train_data[label] = 1
        else: train_data[label] += 1

    for i in result_json['split']['xsub_val']:
        label = str(i.split("_")[-2].replace("B",""))
        if valid_data.get(label) == None: valid_data[label] = 1
        else: valid_data[label] += 1

    print(f"Train data : {train_data}")
    print(f"Validation data : {valid_data}")

def augment_frame(origin):
    result = []
    insert_num = 3
    for i in range(1, len(origin)):
        result.append(list(origin[i-1]))
        for z in range(insert_num):
            output = []
            for j in range(17):
                dist_x = int((origin[i][j][0] - origin[i-1][j][0]) / insert_num)
                dist_y = int((origin[i][j][1] - origin[i-1][j][1]) / insert_num)

                output.append(
                    [int(origin[i-1][j][0] + (dist_x * (z+1))),
                    int(origin[i-1][j][1] + (dist_y * (z+1)))]
                )

            result.append(list(output))
    return result

def calc_rotate(x0, y0, xm, ym, theta):
    x1 = (x0 - xm) * cos(radians(360 - theta)) - (y0 - ym) * sin(radians(360 - theta)) + x0
    y1 = (x0 - xm) * sin(radians(360 - theta)) + (y0 - ym) * cos(radians(360 - theta)) + y0
    return [int(x1), int(y1)]

def convert_keypoint(keypoint):
    trans = [0] * 17

    trans[0] = keypoint[0]
    trans[1] = keypoint[1]
    trans[2] = keypoint[2]
    trans[3] = keypoint[3]
    trans[4] = keypoint[4]
    trans[5] = keypoint[6]
    trans[6] = keypoint[7]
    trans[7] = keypoint[8]
    trans[8] = keypoint[9]
    trans[9] = keypoint[10]
    trans[10] = keypoint[11]
    trans[11] = keypoint[12]
    trans[12] = keypoint[13]
    trans[13] = keypoint[14]
    trans[14] = keypoint[15]
    trans[15] = keypoint[16]
    trans[16] = keypoint[17]

    return trans

def main():
    args = parse_args()

    # convert_pkl()
    convert_dataset(args)

if __name__ == '__main__':
    main()