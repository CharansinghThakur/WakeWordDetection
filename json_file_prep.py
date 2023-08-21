import json
import argparse
import os
import random
import math
from pydub import AudioSegment
from tqdm import tqdm


def main(args):
    zeros = os.listdir(args.zero_label_folder)[:3000]
    ones = os.listdir(args.one_label_folder)

    data = []

    for file in zeros:
        data.append({"audio_path": os.path.join(args.zero_label_folder, file), "label":0})
    for file in ones:
        data.append({"audio_path":os.path.join(args.one_label_folder, file), "label": 1})

    random.shuffle(data)
    data_count = len(data)
    train_split = math.floor(data_count * args.split_percent)

    if not os.path.exists(args.json_dest_folder):
        os.mkdir(args.json_dest_folder)
    with open(os.path.join(args.json_dest_folder,"train_data.json"), 'w') as f:
        for instance in data[:train_split]:
            line = json.dumps(instance)
            f.write(line + "\n")

    with open(os.path.join(args.json_dest_folder,"test_data.json"), 'w') as f:
        for instance in data[train_split:]:
            line = json.dumps(instance)
            f.write(line + '\n')
        
    return "Files are written in {} folder".format(args.json_dest_folder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('''Script for creating train and test json file
                                     from the audio files in seperate folders''')
    
    parser.add_argument('--zero_label_folder', type = str, default= None, required=True)
    parser.add_argument('--one_label_folder', type = str, default= None, required=True)
    parser.add_argument('--json_dest_folder', type = str, default= None, required=True)
    parser.add_argument('--split_percent', type = float, default= None, required=True)
    parser.add_argument('--zero_file_ext', type = str, default = 'wav', required=False)
    parser.add_argument('--one_file_ext', type = str, default = 'wav', required=False)
    args = parser.parse_args()
    main(args)