import os
import sys
import math
import keras 
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras.layers import (Conv2D, 
                          BatchNormalization, 
                          ReLU, 
                          DepthwiseConv2D, 
                          Activation, 
                          Input, 
                          Add, 
                          Flatten, 
                          Dense, 
                          Lambda,
                          Softmax)
                          
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.callbacks import LearningRateScheduler


"""Build MobileFaceNet mdoel"""

def __conv2d_block(_inputs, filters, kernel, strides, is_use_bias=False, padding='same'):
    x = Conv2D(filters, kernel, strides= strides, padding=padding, use_bias=is_use_bias)(_inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def __res_block(_inputs, out_dim, strides, expansion_ratio, is_use_bais=False, shortcut=True):
    # ** to high dim 
    bottleneck_dim = K.int_shape(_inputs)[-1] * expansion_ratio

    # ** pointwise conv 
    x = __conv2d_block(_inputs, bottleneck_dim, kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bais)

    # ** depthwise conv
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # ** pointwise conv
    x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    if shortcut and strides == (1, 1):
        in_dim = K.int_shape(_inputs)[-1]
        if in_dim != out_dim:
            ins = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(_inputs)
            x = Add()([x, ins])
        else:
            x = Add()([x, _inputs])
    return x


def build_model(input_shape=(112, 112, 3), num_classes=24, expansion_ratio=6, embedding_dim=128, s=64, m=0.5):
    image_inputs = Input(shape=input_shape)
    label_inputs = Input(shape=(num_classes,))

    net = __conv2d_block(image_inputs, filters=64, kernel=(3, 3), strides=(2, 2), is_use_bias=False) # size/2 (56)

    net = __res_block(net, out_dim=64, strides=(1, 1), expansion_ratio=1, is_use_bais=False, shortcut=True)

    net = __res_block(net, out_dim=64, strides=(2, 2), expansion_ratio=2, is_use_bais=False, shortcut=True) # size/4 (28)

    net = __res_block(net, out_dim=64, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = __res_block(net, out_dim=64, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = __res_block(net, out_dim=64, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = __res_block(net, out_dim=64, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)

    net = __res_block(net, out_dim=128, strides=(2, 2), expansion_ratio=4, is_use_bais=False, shortcut=True) # size/8 (14)

    net = __res_block(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = __res_block(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = __res_block(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = __res_block(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = __res_block(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = __res_block(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)

    net = __res_block(net, out_dim=128, strides=(2, 2), expansion_ratio=4, is_use_bais=False, shortcut=True) # size/16 (7)

    net = __res_block(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = __res_block(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    
    net = __conv2d_block(net, 512, (1, 1), (1, 1), True, 'valid')
    
    # ** Global Depthwise Conv 
    net = DepthwiseConv2D((7, 7), strides=(1, 1), depth_multiplier=1, padding='valid')(net)
    net = __conv2d_block(net, embedding_dim, (1, 1), (1, 1), True, 'valid')

    # ** embedding layer
    net = Flatten()(net)
    net = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='embeddings')(net)

    # ** loss layer
    net = ArcFaceLossLayer(s=64, m=0.5, num_classes=num_classes, is_use_bais=False)([net, label_inputs])

    # ** build model
    model = Model(inputs=[image_inputs, label_inputs], outputs=net)

    return model

def dummy_loss(y_true, y_pred):
    return y_pred


""" define arc-face loss keras layer """
class ArcFaceLossLayer(Layer):
    def __init__(self, s=64, m=0.5, num_classes=24, is_use_bais=False, **kwargs):
        self.s = float(s)
        self.m = float(m)
        self.num_classes = num_classes
        self.is_use_bais = is_use_bais
        super(ArcFaceLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights = self.add_weight(name='Weights',
                                        shape=(input_shape[0][1], self.num_classes), 
                                        initializer='he_normal',
                                        regularizer=None,
                                        trainable=True,
                                        constraint=None
                                       )
        # input_shape[0][1]
        # shape=(input_shape[0][1], self.num_classes
        super(ArcFaceLossLayer, self).build(input_shape)
         
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

    def call(self, _inputs):
        # ** labels is one-hot format
        embedding_input, labels = _inputs
        s = self.s 
        m = self.m 
        num_classes = self.num_classes 
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        thresh = math.cos(math.pi - m)

        # ** l2 normalization 
        embedding_norm = tf.norm(embedding_input, axis=1, keepdims=True)
        embedding = tf.div(embedding_input, embedding_norm, name='norm_embedding')
        weights_norm = tf.norm(self._weights, axis=0, keepdims=True)
        weights = tf.div(self._weights, weights_norm, name='norm_weights')

        # ** cos(theta + m)
        cos_theta = tf.matmul(embedding, weights, name='cos_theta')
        cos_theta_square = tf.square(cos_theta, name='cos_theta_square')
        sin_theta_square = tf.subtract(1.0, cos_theta_square, name='sin_theta_square')
        sin_theta = tf.sqrt(sin_theta_square, name='sin_theta')
        cos_theta_m = s * tf.subtract(tf.multiply(cos_theta, cos_m), tf.multiply(sin_theta, sin_m), name='cos_theta_m')

        # ** this condition controls the theta+m should in range [0, pi]
        # **  0 <= theta + m <= pi
        # ** -m <= theta + 0 <= pi - m
        cond_value = cos_theta - thresh 
        cond_index = tf.cast(tf.nn.relu(cond_value, name='if_else'), dtype=tf.bool)
        keep_value = s * (cos_theta - sin_m * m)
        cos_theta_m_adj = tf.where(cond_index, cos_theta_m, keep_value)
        s_cos_theta = tf.multiply(s, cos_theta, name='scalar_cos_theta')

        # **
        mask = labels
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        # ** logit
        logit = tf.add(tf.multiply(s_cos_theta, inv_mask), tf.multiply(cos_theta_m_adj, mask), name='arcface_loss_output')
        
        # ** loss
        self.inference_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=labels))

        return self.inference_loss

def lr_scheduler(epoch):
    if epoch > 50:
        lr = 1e-3
    elif epoch > 100:
        lr = 5e-4
    elif epoch > 300:
        lr = 1e-4
    elif epoch > 500:
        lr = 5e-5
    elif epoch > 800:
        lr = 1e-5
    else:
        lr = 3e-3
    print('learning rate : {}'.format(lr))
    return lr

if __name__ == '__main__':
    model = build_model()
    optmizer = Adam(lr=1e-4)
    model.compile(optimizer='adam', loss=dummy_loss)
    print('\n\n')
    print('summary')
    print(model.summary())








