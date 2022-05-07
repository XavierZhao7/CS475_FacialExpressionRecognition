import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from skimage.filters import gabor_kernel
from tensorflow.keras import backend as K
import cv2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dropout, Conv2DTranspose, LeakyReLU, Concatenate, Lambda

# Method one

class GaborFilterBank(layers.Layer):
    def __init__(self):
        super().__init__()
    
    def build(self, input_shape):
        # assumption: shape is NHWC 
        self.n_channel = input_shape[-1]
        self.kernels = []
        for theta in range(4):
            theta = theta / 4.0 * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.15, 0.25, 0.35):
                    kernel = np.real(
                        gabor_kernel(
                            frequency, theta=theta, sigma_x=sigma, sigma_y=sigma
                        )
                    ).astype(np.float32)
                    # tf.nn.conv2d does crosscorrelation, needs flip
                    kernel = np.flip(kernel)
                    # match the number of channel of the input
                    kernel = np.stack((kernel,)*self.n_channel, axis=-1)
                    # adding the number of out channel : 1
                    kernel = kernel[:, :, : , np.newaxis] 
                    self.kernels.append(tf.Variable(kernel, trainable=False))

    def call(self, x):
        out_list = []
        for kernel in self.kernels:
            out_list.append(tf.nn.conv2d(x, kernel, strides=1, padding="SAME"))
        # output is [batch_size, H, W, 32] 
        # where 32 is the number of filters 
        # ga32 = n_theta * n_sigma * n_freq = 4 * 2 * 4 
        return tf.concat(out_list,axis=-1)




