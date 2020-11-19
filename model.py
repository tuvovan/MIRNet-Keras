import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import Conv2D, ReLU, Concatenate, Add, GlobalAveragePooling2D, UpSampling2D, Softmax, MaxPooling2D
from tensorflow.keras.activations import sigmoid
from utils import *

class MIRNet(Model):
    def __init__(self, channels, num_mrb, num_rrg):
        super(MIRNet, self).__init__()
        self.channels = channels
        self.num_mrb = num_mrb
        self.num_rrg = num_rrg

    def SKFF(self, L1, L2, L3):
        c = list(L1.shape)[-1]
        L = Add()([L1, L2, L3])
        gap = GlobalAveragePooling2D()(L)
        S = tf.reshape(gap, shape=(-1,1,1,c))
        Z = ReLU()(Conv2D(filters=c//8, kernel_size=(1,1))(S))

        # parallel 1
        v1 = Softmax()(Conv2D(c, kernel_size=(1,1))(Z))
        L1 = L1 * v1
        # parallel 2
        v2 = Softmax()(Conv2D(c, kernel_size=(1,1))(Z))
        L2 = L2 * v2
        # parallel 3
        v3 = Softmax()(Conv2D(c, kernel_size=(1,1))(Z))
        L3 = L3 * v3

        U = Add()([L1, L2, L3])

        return U

    def CA(self, X):
        c = list(X.shape)[-1]
        gap = GlobalAveragePooling2D()(X)
        d = tf.reshape(gap, shape=(-1,1,1,c))
        d1 = ReLU()(Conv2D(filters=c//8, kernel_size=(1,1))(d))
        d_bid = sigmoid(Conv2D(filters=c, kernel_size=(1,1))(d1))

        return X*d_bid

    def SA(self, X):
        gap = tf.reduce_max(X, axis=-1)
        gap = tf.expand_dims(gap, axis=-1)
        gmp = tf.reduce_mean(X, axis=-1)
        gmp = tf.expand_dims(gmp, axis=-1)
        
        ff = Concatenate(axis=-1)([gap, gmp])

        f = Conv2D(1, kernel_size=(1,1))(ff)
        f = sigmoid(f)

        return X * f

    def DAU(self, X):
        c = list(X.shape)[-1]
        M = Conv2D(c, kernel_size=(3,3), padding='same')(X)
        M = ReLU()(M)
        M = Conv2D(c, kernel_size=(3,3), padding='same')(M)

        ca = self.CA(M)
        sa = self.SA(M)

        concat = Concatenate(axis=-1)([ca, sa])

        concat2 = Conv2D(c, kernel_size=(1,1))(concat)

        return Add()([X, concat2])


    def DownSampling(self, X):
        c = list(X.shape)[-1]
        upper_branch = Conv2D(c, kernel_size=(1,1))(X)
        upper_branch = ReLU()(upper_branch)

        upper_branch = Conv2D(c, kernel_size=(3,3), padding='same')(X)
        upper_branch = ReLU()(upper_branch)
        upper_branch = MaxPooling2D()(upper_branch)
        upper_branch = Conv2D(c * 2, kernel_size=(1,1))(upper_branch)

        lower_branch = MaxPooling2D()(X)
        lower_branch = Conv2D(c * 2, kernel_size=(1,1))(lower_branch)

        return Add()([lower_branch, upper_branch])

    def UpSampling(self, X):
        c = list(X.shape)[-1]
        upper_branch = Conv2D(c, kernel_size=(1,1))(X)
        upper_branch = ReLU()(upper_branch)

        upper_branch = Conv2D(c, kernel_size=(3,3), padding='same')(X)
        upper_branch = ReLU()(upper_branch)
        upper_branch = UpSampling2D()(upper_branch)
        upper_branch = Conv2D(c // 2, kernel_size=(1,1))(upper_branch)

        lower_branch = UpSampling2D()(X)
        lower_branch = Conv2D(c // 2, kernel_size=(1,1))(lower_branch)

        return Add()([lower_branch, upper_branch])


    def MRB(self, X):
        # features
        level1 = X
        level2 = self.DownSampling(X)
        level3 = self.DownSampling(level2)

        # DAU
        level1_DAU = self.DAU(level1)
        level2_DAU = self.DAU(level2)
        level3_DAU = self.DAU(level3)

        # SKFF
        level1_SKFF = self.SKFF(level1_DAU, self.UpSampling(level2_DAU), self.UpSampling(self.UpSampling(level3_DAU)))
        level2_SKFF = self.SKFF(self.DownSampling(level1_DAU), level2_DAU, self.UpSampling(level3_DAU))
        level3_SKFF = self.SKFF(self.DownSampling(self.DownSampling(level1_DAU)), self.DownSampling(level2_DAU), level3_DAU)

        # DAU 2
        level1_DAU_2 = self.DAU(level1_SKFF)
        level2_DAU_2 = self.UpSampling((self.DAU(level2_SKFF)))
        level3_DAU_2 = self.UpSampling(self.UpSampling(self.DAU(level3_SKFF)))

        # SKFF 2
        SKFF_ = self.SKFF(level1_DAU_2, level3_DAU_2, level3_DAU_2)

        conv = Conv2D(self.channels, kernel_size=(3,3), padding='same')(SKFF_)

        return Add()([X, conv])

    def RRG(self, X, num_mrb):
        conv1 = Conv2D(self.channels, kernel_size=(3,3), padding='same')(X)
        for _ in range(num_mrb):
            conv1 = self.MRB(conv1)
        
        conv2 = Conv2D(self.channels, kernel_size=(3,3), padding='same')(conv1)

        return Add()([conv2, X])
        

    def main_model(self, X):
        X1 = Conv2D(self.channels, kernel_size=(3,3), padding='same')(X)
        
        for _ in range(self.num_rrg):
            X1 = self.RRG(X1, self.num_mrb)
        
        conv = Conv2D(3, kernel_size=(3,3), padding='same')(X1)
        output = Add()([X, conv])
        
        return output



class BlurPool2D():
    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(BlurPool2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 5:
            bk = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
            bk = bk / np.sum(bk)
        else:
            raise ValueError

        bk = np.repeat(bk, input_shape[3])

        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[3], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[3], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(BlurPool2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]