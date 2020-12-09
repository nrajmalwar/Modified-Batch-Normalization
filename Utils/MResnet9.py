import numpy as np
import os
import time
import datetime
import math

# tensorflow imports
import tensorflow as tf
import tensorflow.keras.backend as K

# tf.keras imports
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Add, Softmax, GlobalMaxPool2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Input, ReLU, Lambda, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from MBatchNorm import BatchNorm

# PyTorch’s default way to set the initial, random weights of layers does not have 
# a counterpart in Tensorflow, so we borrow that
def init_pytorch(shape, dtype=tf.float32, partition_info=None):
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

def Resnet9(input_dim, n_classes, hparams, stride=1): 
    inputs = Input(shape= input_dim, name='input')

    # Define the repeated conv_bn_relu function
    def conv_bn_relu(x, layer_name, pool=False):
        channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 256}
        x = Conv2D(filters = channels[layer_name], 
                   kernel_size = 3, 
                   strides = 1, 
                   padding = 'same',
                   kernel_initializer=init_pytorch,
                   kernel_regularizer=l2(hparams['decay']),
                   use_bias=False)(x)

        # Apply BatchNorm              
        x = BatchNorm(momentum=0.9, epsilon=1e-5, stride=stride)(x)
        
        x = ReLU()(x)
        if pool==True:
            x = MaxPooling2D(2)(x)
        return x        

    # Stack the layers
    # Layer Prep
    layer_prep = conv_bn_relu(inputs, 'prep')

    # Layer 1
    layer1_part1 = conv_bn_relu(layer_prep, 'layer1', pool=True)  
    layer1_part2_res1 = conv_bn_relu(layer1_part1, 'layer1')
    layer1_part2_res2 = conv_bn_relu(layer1_part2_res1, 'layer1')
    layer1 = Add()([layer1_part1, layer1_part2_res2])
    
    # Layer 2
    layer2_part1 = conv_bn_relu(layer1, 'layer2', pool=True)  
    layer2_part2_res1 = conv_bn_relu(layer2_part1, 'layer2')
    layer2_part2_res2 = conv_bn_relu(layer2_part2_res1, 'layer2')
    layer2 = Add()([layer2_part1, layer2_part2_res2])

    # Layer 3
    layer3 = conv_bn_relu(layer2, 'layer3', pool=True)

    # Layer Classifier
    layer_classifier = GlobalMaxPool2D()(layer3)
    layer_classifier = Flatten()(layer_classifier)
    layer_classifier = Dense(n_classes, kernel_initializer= init_pytorch,
                           kernel_regularizer=l2(hparams['decay']),
                           use_bias=False, name='classifier')(layer_classifier)
    layer_classifier = Lambda(lambda x : x * 0.2)(layer_classifier)

    outputs = Softmax()(layer_classifier)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr= hparams['learning_rate'], momentum= hparams['momentum'], 
                              nesterov=True),
                metrics=['accuracy'])
    return model