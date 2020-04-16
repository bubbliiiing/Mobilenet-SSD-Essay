import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten,Add,ZeroPadding2D
from keras.layers import GlobalAveragePooling2D,DepthwiseConv2D,BatchNormalization
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import merge, concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=1,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def relu6(x):
    return K.relu(x, max_value=6)


def conv2d_bn(x,filters,num_row,num_col,padding='same',stride=1,dilation_rate=1,relu=True):
    x = Conv2D(
        filters, (num_row, num_col),
        strides=(stride,stride),
        padding=padding,
        dilation_rate=(dilation_rate, dilation_rate),
        use_bias=False)(x)
    x = BatchNormalization()(x)
    if relu:    
        x = Activation(relu6)(x)
    return x

def mobilenet(input_tensor):
    #----------------------------主干特征提取网络开始---------------------------#
    # SSD结构,net字典
    net = {} 
    # Block 1
    x = input_tensor
    # 300,300,3 -> 150,150,64
    x = Conv2D(32, (3,3),
            padding='same',
            use_bias=False,
            strides=(2, 2),
            name='conv1')(input_tensor)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)
    x = _depthwise_conv_block(x, 64, 1, block_id=1)
    
    # 150,150,64 -> 75,75,128
    x = _depthwise_conv_block(x, 128, 1,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, 1, block_id=3)

    
    # Block 3
    # 75,75,128 -> 38,38,256
    x = _depthwise_conv_block(x, 256, 1,
                              strides=(2, 2), block_id=4)
    
    x = _depthwise_conv_block(x, 256, 1, block_id=5)

    # Block 4
    # 38,38,256 -> 19,19,512
    x = _depthwise_conv_block(x, 512, 1,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, 1, block_id=7)
    x = _depthwise_conv_block(x, 512, 1, block_id=8)
    x = _depthwise_conv_block(x, 512, 1, block_id=9)
    x = _depthwise_conv_block(x, 512, 1, block_id=10)
    x = _depthwise_conv_block(x, 512, 1, block_id=11)
    net['conv4_3'] = x

    # Block 5
    # 19,19,512 -> 10,10,1024
    x = _depthwise_conv_block(x, 1024, 1,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, 1, block_id=13)
    net['fc7'] = x

    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    # 10,10,512 -> 5,5,512
    net['conv6_1'] = conv2d_bn(net['fc7'], 256, 1, 1)
    net['conv6_2'] = conv2d_bn(net['conv6_1'], 512, 3, 3, stride=2)

    # Block 7
    # 5,5,512 -> 3,3,256
    net['conv7_1'] = conv2d_bn(net['conv6_2'], 128, 1, 1)
    net['conv7_2'] = conv2d_bn(net['conv7_1'], 256, 3, 3, stride=2)

    # Block 8
    # 3,3,256 -> 2,2,256
    net['conv8_1'] = conv2d_bn(net['conv7_2'], 128, 1, 1)
    net['conv8_2'] = conv2d_bn(net['conv8_1'], 256, 3, 3, stride=2)

    # Block 9
    # 3,3,256 -> 1,1,256
    net['conv9_1'] = conv2d_bn(net['conv8_2'], 64, 1, 1)
    net['conv9_2'] = conv2d_bn(net['conv9_1'], 128, 3, 3, stride=2)
    #----------------------------主干特征提取网络结束---------------------------#
    return net