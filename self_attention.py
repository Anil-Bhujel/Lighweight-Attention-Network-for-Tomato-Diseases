# Self-attention module
import tensorflow as tf
from tensorflow.keras.layers import Add, Activation, Conv2D

def attention(x, channels):
    batch_size, height, width, num_channels = x.get_shape().as_list()
    f = Conv2D(channels // 8, kernel_size=(1,1), strides=(1,1), padding = 'same', use_bias = True)(x) # [bs, h, w, c']
    f = Activation('relu')(f)
    g = Conv2D(channels // 8, kernel_size=(1,1), strides=(1,1), padding = 'same', use_bias = True)(x) # [bs, h, w, c']
    g = Activation('relu')(g)
    h = Conv2D(channels, kernel_size=(1,1), strides=(1,1), padding = 'same', use_bias = True)(x) # [bs, h, w, c]
    h = Activation('relu')(h)
    print(f.shape)
    print(g.shape)
    print(h.shape)
#     f=hw_flatten(f)

    area = height*width
    f = tf.reshape(f,shape = [-1, area, f.shape[-1]])
    g = tf.reshape(g,shape = [-1, area, g.shape[-1]])
    h = tf.reshape(h,shape = [-1, area, h.shape[-1]])
    
#     print(f.shape)
#     print(hw_flatten(g).shape)
    

    # N = h * w
#     s = tf.matmul(hw_flatten(f), hw_flatten(g), transpose_b=True) # # [bs, N, N]

    s = tf.matmul(g, f, transpose_b=True)
    beta = tf.keras.activations.softmax(s, axis=-1)  # attention map

    SA = tf.matmul(beta, h) # [bs, N, C]
    gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    SA = tf.reshape(SA, shape = [-1,height, width,num_channels]) # [bs, h, w, C]
    SA = Conv2D(channels, kernel_size=(1,1), strides=(1,1), padding = 'same', name = 'self_attention')(SA)

#     out = gamma * SA + x
    out = Add()([gamma*SA, x])
    print(out.shape)

    return out