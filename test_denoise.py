import os
import cv2
import sys
import glob
import time
import math
import argparse
import numpy as np
import tensorflow as tf 

from model import *
from utils import *
from tensorflow.keras import Model, Input


def run(config, model):
    for name in os.listdir(config.test_path):
        fullname = os.path.join(config.test_path, name)
        lr = cv2.imread(fullname)
        ft = cv2.imread(fullname.replace('test', 'gt').replace('NOISY', 'GT'))

        lr = np.array(get_lowres_image(PIL.Image.fromarray(lr), mode='denoise'))
        ft = np.array(get_lowres_image(PIL.Image.fromarray(ft), mode='denoise'))

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        ft = cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
        out = predict_images(model, lr)
        print(name, ' --- PSNR: ', psnr_denoise(ft, np.array(out)))
        cv2.imwrite(os.path.join(fullname.replace('test', 'result')), cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

	# Input Parameters
    parser.add_argument('--test_path', type=str, default="test/denoise/")
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--checkpoint_filepath', type=str, default="weights/denoise/")
    parser.add_argument('--num_rrg', type=int, default= 3)
    parser.add_argument('--num_mrb', type=int, default= 2)
    parser.add_argument('--num_channels', type=int, default= 64)

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    mri_x = MIRNet(config.num_channels, config.num_mrb, config.num_rrg)
    x = Input(shape=(None, None, 3))
    out = mri_x.main_model(x)
    model = Model(inputs=x, outputs=out)
    model.summary()
    model.load_weights(config.checkpoint_filepath + '47_36.24.h5')

    run(config, model)