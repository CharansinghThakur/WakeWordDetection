import os
from tqdm import tqdm
from pydub import AudioSegment
import argparse

def mp3_to_wavs(src_dir):
    os.mkdir(src_dir+'_wavs')
    files = os.listdir(src_dir)
    for file in tqdm(files):
        file_path = os.path.join(src_dir, file)
        sound = AudioSegment.from_mp3(file_path)
        sound.export(
            os.path.join(src_dir+'_wavs', os.path.splitext(file)[0]+'.wav'), 
            format="wav")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_dir', type = str, required=True)

    args = parser.parse_args()
    mp3_to_wavs(args.src_dir)
