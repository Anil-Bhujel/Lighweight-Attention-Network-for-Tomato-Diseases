from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid

# from keras.layers import Activation, Conv2D
# import keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import add
from dual_attention import PAM, CAM


# AttributeError: 'KerasTensor' object has no attribute '_keras_shape' change ._keras_shape to .shape
# Attention modules attachment to the main network
def attach_attention_module(net, attention_module):
    if attention_module == 'se_block': # SE_block
        net = se_block(net)
    elif attention_module == 'cbam_block': # CBAM_block
        net = cbam_block(net)
    elif attention_module == 'ca_block': # CA_block
        net = ca_block(net)
    elif attention_module == 'sa_block': # SA_block
        net = sa_block(net)
    elif attention_module == 'da_block': # Dual_block
        net = da_block(net)
    elif attention_module == 'da_module': # Dual_block
        net = da_module(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net

# Squeeze-and-excitation attention module
def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature

# CBAM attention
def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

# CA attention
def ca_block(ca_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    ca_feature = channel_attention(ca_feature, ratio)
    return ca_feature

# Channel attention module
def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

# Spatial attention module
def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])
        

# Self-attention module
def sa_block(x, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channels = x.shape[channel_axis]
    batch_size, height, width, num_channels = x.get_shape().as_list()
    f = Conv2D(channels // ratio, kernel_size=(1,1), strides=(1,1), padding = 'same', use_bias = True)(x) # [bs, h, w, c']
    f = Activation('relu')(f)
    g = Conv2D(channels // ratio, kernel_size=(1,1), strides=(1,1), padding = 'same', use_bias = True)(x) # [bs, h, w, c']
    g = Activation('relu')(g)
    h = Conv2D(channels, kernel_size=(1,1), strides=(1,1), padding = 'same', use_bias = True)(x) # [bs, h, w, c]
    h = Activation('relu')(h)
#     print(f.shape)
#     print(g.shape)
#     print(h.shape)
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

# Dual attention modules 
def da_layer(net_layer):
    pam = PAM()(net_layer)
    cam = CAM()(net_layer)
    feature_sum = add([pam, cam])
    return feature_sum

# Dual attention modules in network
def da_module(da_feature):
    pam = pa_module(da_feature)
    cam = ca_module(da_feature)
    feature_sum = add([pam, cam])
    return feature_sum

# Pixel attention module
def pa_module(input_feature):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channels = input_feature.shape[channel_axis]
    input_shape = input_feature.get_shape().as_list()
    _, h, w, filters = input_shape
    
    b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input_feature)
    c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input_feature)
    d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(input_feature)

    vec_b = K.reshape(b, (-1, h * w, filters // 8))
    vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
    bcT = K.batch_dot(vec_b, vec_cT)
    softmax_bcT = Activation('softmax')(bcT)
    vec_d = K.reshape(d, (-1, h * w, filters))
    bcTd = K.batch_dot(softmax_bcT, vec_d)
    bcTd = K.reshape(bcTd, (-1, h, w, filters))
    
    gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
    out = gamma*bcTd + input_feature
    return out

# Channel attention module
def ca_module(input_feature):
    input_shape = input_feature.get_shape().as_list()
    _, h, w, filters = input_shape

    vec_a = K.reshape(input_feature, (-1, h * w, filters))
    vec_aT = tf.transpose(vec_a, (0, 2, 1))
    aTa = K.batch_dot(vec_aT, vec_a)
    softmax_aTa = Activation('softmax')(aTa)
    aaTa = K.batch_dot(vec_a, softmax_aTa)
    aaTa = K.reshape(aaTa, (-1, h, w, filters))

    gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
    out = gamma*aaTa + input_feature
    return out

# class PAM(Layer):
#     def __init__(self,
#                  gamma_initializer=tf.zeros_initializer(),
#                  gamma_regularizer=None,
#                  gamma_constraint=None,
#                  **kwargs):
#         super(PAM, self).__init__(**kwargs)
#         self.gamma_initializer = gamma_initializer
#         self.gamma_regularizer = gamma_regularizer
#         self.gamma_constraint = gamma_constraint

#     def build(self, input_shape):
#         self.gamma = self.add_weight(shape=(1, ),
#                                      initializer=self.gamma_initializer,
#                                      name='gamma',
#                                      regularizer=self.gamma_regularizer,
#                                      constraint=self.gamma_constraint)

#         self.built = True

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def call(self, input):
#         input_shape = input.get_shape().as_list()
#         _, h, w, filters = input_shape
        
#         initializer = tf.keras.initializers.HeNormal() # added later 
        
#         b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer=initializer)(input)
#         c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer=initializer)(input)
#         d = Conv2D(filters, 1, use_bias=False, kernel_initializer=initializer)(input)
        
#         #b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
#         #c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
#         #d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(input)

#         vec_b = K.reshape(b, (-1, h * w, filters // 8))
#         vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
#         bcT = K.batch_dot(vec_b, vec_cT)
#         softmax_bcT = Activation('softmax')(bcT)
#         vec_d = K.reshape(d, (-1, h * w, filters))
#         bcTd = K.batch_dot(softmax_bcT, vec_d)
#         bcTd = K.reshape(bcTd, (-1, h, w, filters))

#         out = self.gamma*bcTd + input
#         return out


# class CAM(Layer):
#     def __init__(self,
#                  gamma_initializer=tf.zeros_initializer(),
#                  gamma_regularizer=None,
#                  gamma_constraint=None,
#                  **kwargs):
#         super(CAM, self).__init__(**kwargs)
#         self.gamma_initializer = gamma_initializer
#         self.gamma_regularizer = gamma_regularizer
#         self.gamma_constraint = gamma_constraint

#     def build(self, input_shape):
#         self.gamma = self.add_weight(shape=(1, ),
#                                      initializer=self.gamma_initializer,
#                                      name='gamma',
#                                      regularizer=self.gamma_regularizer,
#                                      constraint=self.gamma_constraint)

#         self.built = True

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def call(self, input):
#         input_shape = input.get_shape().as_list()
#         _, h, w, filters = input_shape

#         vec_a = K.reshape(input, (-1, h * w, filters))
#         vec_aT = tf.transpose(vec_a, (0, 2, 1))
#         aTa = K.batch_dot(vec_aT, vec_a)
#         softmax_aTa = Activation('softmax')(aTa)
#         aaTa = K.batch_dot(vec_a, softmax_aTa)
#         aaTa = K.reshape(aaTa, (-1, h, w, filters))

#         out = self.gamma*aaTa + input
#         return out
    
