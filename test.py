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

        lr = np.array(get_lowres_image(PIL.Image.fromarray(lr)))
        ft = np.array(get_lowres_image(PIL.Image.fromarray(ft)))

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        ft = cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
        out = predict_images(model, lr)
        # prefix = fullname[:-4]
        # title = 'test'
        # plot_results(out, prefix, title)
        print(name, ' --- PSNR: ', psnr(ft, np.array(out)))
        cv2.imwrite(os.path.join(fullname.replace('.PNG', '_out.PNG')), cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

	# Input Parameters
    parser.add_argument('--test_path', type=str, default="test/")
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--weight_test_path', type=str, default= "weights/best.h5")

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    rfanet_x = MIRNet(64, 2, 3)
    x = Input(shape=(None, None, 3))
    out = rfanet_x.main_model(x)
    model = Model(inputs=x, outputs=out)
    model.summary()
    model.load_weights(config.weight_test_path)

    run(config, model)