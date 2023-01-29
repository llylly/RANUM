"""
This script copies files other than .pt from results/ folder
Since .pt files are too large and does not contain useful result statistics
"""
import argparse
import shutil
import os

cnt = 0

def copy_worker(source, dest):
    global cnt
    for file in os.listdir(source):
        if os.path.isdir(os.path.join(source, file)):
            if not os.path.exists(os.path.join(dest, file)):
                os.makedirs(os.path.join(dest, file))
            copy_worker(os.path.join(source, file), os.path.join(dest, file))
        else:
            if not file.endswith('.pt'):
                cnt += 1
                print(f'[{cnt}] copy {os.path.join(source, file)} to {os.path.join(dest, file)}')
                shutil.copy(os.path.join(source, file), os.path.join(dest, file))


parser = argparse.ArgumentParser()
parser.add_argument('source', type=str)
parser.add_argument('dest', type=str)
if __name__ == '__main__':
    args = parser.parse_args()
    copy_worker(args.source, args.dest)
