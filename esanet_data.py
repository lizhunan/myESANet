import argparse
from src.datasets.prepare_datasets import Preparation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    parser.add_argument('--nyuv2-dir', type=str, default=None, required=True, help='filepath to NYUv2 mat file')
    parser.add_argument('--nyuv2-output-path', type=str, required=True, help='path where to store dataset')
    parser.add_argument('--download', type=bool,default=True, help='path where to store dataset')
    arg = parser.parse_args()

    preparation = Preparation(arg)
    preparation()