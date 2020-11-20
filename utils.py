import os
import cv2
import time
import math
import tensorflow as tf
import numpy as np

from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow import keras


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array


def plot_results(img, prefix, title, mode):
    """Plot the result with zoom-in area."""
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    if mode == 'denoise':

        # Create a new figure with a default 111 subplot.
        fig, ax = plt.subplots()
        im = ax.imshow(img_array[::-1], origin="lower")

        plt.title(title)
        # zoom-factor: 2.0, location: upper-left
        axins = zoomed_inset_axes(ax, 2, loc=2)
        axins.imshow(img_array[::-1], origin="lower")

        # Specify the limits.
        x1, x2, y1, y2 = 200, 350, 100, 250
        # Apply the x-limits.
        axins.set_xlim(x1, x2)
        # Apply the y-limits.
        axins.set_ylim(y1, y2)

        plt.yticks(visible=False)
        plt.xticks(visible=False)

        # Make the line.
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
        plt.savefig(str(prefix) + "-" + title + ".png")
    elif mode == 'delight':
        cv2.imwrite(os.path.join(str(prefix) + "-" + title + ".png"), cv2.cvtColor(np.uint8(img_to_array(img)), cv2.COLOR_BGR2RGB))
    


def get_lowres_image(img, mode):
    """Return noisy image to use as model input."""
    if mode == 'denoise':
        size = (1024, 720)
        img = img.resize(size)
    elif mode == 'delight':
        img = img
    return img


def predict_images(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    y = img_to_array(img)
    # y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    t = time.time()
    out = model.predict(input)
    print(time.time() - t)

    out_img_y = out[0]
    # out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1], 3))
    out_img = PIL.Image.fromarray(np.uint8(out_img_y))
    return out_img

class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self, test_img_paths, mode, checkpoint_ep):
        super(ESPCNCallback, self).__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), mode=mode)
        self.mode = mode
        self.checkpoint_ep = checkpoint_ep

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if (epoch + 1)  % self.checkpoint_ep == 0:
            prediction = predict_images(self.model, self.test_img)
            plot_results(prediction, "epoch-" + str(epoch), "prediction", mode=self.mode)

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(255.0 / logs["loss"]))


class SSID:
    def __init__(self,
                 subset='train', instant_name='SIDD_Small_sRGB_Only/Scene_Instances.txt',
                 images_dir='SIDD_Small_sRGB_Only/Data'):
        
        with open(instant_name) as f:
            ls = [l.rstrip() for l in f]

        if subset == 'train':
            self.image_ids = range(0, 120)
            self.data_ids = [ls[i] for i in self.image_ids]
        elif subset == 'valid':
            self.image_ids = range(120, 155)
            self.data_ids = [ls[i] for i in self.image_ids]
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.subset = subset
        self.images_dir = images_dir

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=4, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
            ds = ds.map(scaling, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        ds = self._images_dataset(self._hr_image_files())
        return ds

    def lr_dataset(self):
        ds = self._images_dataset(self._lr_image_files())
        return ds

    def _hr_image_files(self):
        return [os.path.join(self.images_dir, f'{image_id}', 'GT_SRGB_010.PNG') for image_id in self.data_ids]

    def _lr_image_files(self):
        return [os.path.join(self.images_dir, f'{image_id}', 'NOISY_SRGB_010.PNG') for image_id in self.data_ids]

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')


class LOL:
    def __init__(self,
                 subset='train'):

        if subset == 'train':
            self.images_dir = "our485"
            self.data_ids = [i for i in sorted(os.listdir(os.path.join(self.images_dir, 'high'))) if 'png' in i]
        elif subset == 'valid':
            self.images_dir = "eval15"
            self.data_ids = [i for i in sorted(os.listdir(os.path.join(self.images_dir, 'high'))) if 'png' in i]
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.subset = subset
        # self.images_dir = images_dir

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=4, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
            # ds = ds.map(scaling, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        ds = self._images_dataset(self._hr_image_files())
        return ds

    def lr_dataset(self):
        ds = self._images_dataset(self._lr_image_files())
        return ds

    def _hr_image_files(self):
        return [os.path.join(self.images_dir, 'high', f'{image_id}') for image_id in self.data_ids]

    def _lr_image_files(self):
        return [os.path.join(self.images_dir, 'low', f'{image_id}') for image_id in self.data_ids]

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------


def random_crop(lr_img, hr_img, hr_crop_size=128):
    lr_crop_size = hr_crop_size
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w
    hr_h = lr_h

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)

def scaling(lr_img, hr_img):
    lr_img = tf.cast(lr_img, tf.float32)
    hr_img = tf.cast(hr_img, tf.float32)
    lr_img = lr_img / 255.0
    hr_img = hr_img / 255.0
    return lr_img, hr_img


# -----------------------------------------------------------
#  IO
# -----------------------------------------------------------

def psnr_denoise(y_true, y_pred):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    return tf.image.psnr(y_pred, y_true, max_val=1.0)

def psnr_delight(y_true, y_pred):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


def custom_loss_function(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred) +  tf.square(1e-3)
    return tf.sqrt(tf.reduce_mean(squared_difference, axis=-1))
    