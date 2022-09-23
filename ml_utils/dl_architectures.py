
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers



def set_Conv3d_ConvLstm_model(width=150, height=150, depth = 5, channels = 8, initfilters = 64):
    """Build a 3D convolutional neural network model."""
    #data_format="NDHWC"
    #inputs = keras.Input((width, height, depth, 3))
    kernel_sizeinit = (depth-1, 2, 2)
    kernel_sizesecond = (depth-2, 3, 3)


    inputs3d = keras.Input((depth, width, height,channels))

    x = layers.Conv3D(filters=initfilters, kernel_size=kernel_sizeinit, strides=(1,2,2), 
                      padding='same', activation="relu",data_format='channels_last')(inputs3d)

    x = layers.MaxPool3D(pool_size=kernel_sizeinit, strides=(1,2,2), padding='same',data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv3D(filters=initfilters*2, kernel_size=kernel_sizesecond, strides=(1,2,2), 
                    padding='same',  activation="relu",data_format='channels_last')(x)
    x = layers.MaxPool3D(pool_size=kernel_sizesecond, strides=(1,2,2), padding='same',data_format='channels_last')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.ConvLSTM2D(
        filters=initfilters*2,
        kernel_size=(5, 5),
        data_format='channels_last',
        padding="same",
        return_sequences=True,
        activation="relu",
        )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        data_format='channels_last',
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        data_format='channels_last',
        activation="relu",
    )(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv3D(
        filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same", strides=2
    )(x)
    #x = layers.MaxPool3D(pool_size=kernel_sizeinit, 
    #                    strides=2, padding='same',data_format='channels_last')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(units=2056, activation="relu",kernel_regularizer=tf.keras.regularizers.L1(0.01))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(units=1028, activation="relu",kernel_regularizer=tf.keras.regularizers.L1(0.01),)(x) 
    
    return keras.Model(inputs3d, x, name="3dcnnlstm")



def ConvLSTM_Model(frames, channels, width, height, initfilters = 128):
  
    inputs  = keras.Input(shape=(frames, width, height,channels))
    
    first_ConvLSTM = layers.ConvLSTM2D(filters=initfilters, kernel_size=(2, 2)
                       , data_format='channels_last'
                       , activation='relu'
                       #, recurrent_activation='hard_sigmoid'
                       , padding='same', return_sequences=True)(inputs)

    first_batchnorm = layers.BatchNormalization()(first_ConvLSTM)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=1, padding='same')(first_batchnorm)
    x = layers.Dropout(0.5)(x)
    second_ConvLSTM = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3)
                        , data_format='channels_last'
                        , activation='relu'
                        , padding='same', return_sequences=True)(x)
    second_batchnorm = layers.BatchNormalization()(second_ConvLSTM)
    x = tf.keras.layers.MaxPool3D(pool_size=(3,3,3), strides=2, padding='same')(second_batchnorm)
    x = layers.Dropout(0.5)(x)
    out_shape = x.shape
    print('====Model shape: ', out_shape)
    third_ConvLSTM = layers.ConvLSTM2D(filters=16, kernel_size=(5, 5)
                        , data_format='channels_last'
                        , activation='relu'
                        , padding='same', return_sequences=True)(x)

    batchnorm = layers.BatchNormalization()(third_ConvLSTM)
    x = tf.keras.layers.MaxPool3D(pool_size=(5,5,5), strides=2, padding='same')(batchnorm)
    
    x = layers.Dropout(0.5)(x)
    out_shape = x.shape
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(0.5)(x)
    out_shape = x.shape
    print('====Model shape: ', out_shape)
    #reshaped = layers.Reshape((out_shape[1], out_shape[2] * out_shape[3] * out_shape[4]))(x)

#    x = layers.LSTM(128, return_sequences=False,kernel_regularizer=tf.keras.regularizers.L1(0.01),
#                                                   activity_regularizer=tf.keras.regularizers.L2(0.01))(reshaped)
    #x = layers.LSTM(128, return_sequences=False)(reshaped)

    #x = layers.Dropout(0.5)(second_BatchNormalization)

    #x = layers.Conv3D(
    #    filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same", 
    #    strides=2
        
    #)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.GlobalAveragePooling3D()(x)
    #x = layers.Dropout(0.5)(x)
    #x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.Flatten()(x)
    out_shape = x.shape
    print('====Model shape: ', out_shape)

    #x = layers.Dense(units=2056, activation="relu")(x)
    #x = layers.Dense(units=1028, activation="linear")(x)

    x = layers.Dense(units=512, activation="linear",
                    kernel_regularizer=tf.keras.regularizers.L1(0.01))(x)

    x = layers.Dense(units=256, activation="linear")(x)

    x = layers.Dropout(0.5)(x)

    return keras.Model(inputs, x, name="convlstm")

def ConvLSTM_Model_v2(frames, channels, width, height):
  
    inputs  = keras.Input(shape=(frames, width, height,channels))
    
    first_ConvLSTM = layers.ConvLSTM2D(filters=64, kernel_size=(2, 2)
                       , data_format='channels_last'
                       , activation='relu'
                       #, recurrent_activation='hard_sigmoid'
                       , padding='same', return_sequences=True)(inputs)

    first_BatchNormalization = layers.BatchNormalization()(first_ConvLSTM)
    #x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=2, padding='same')(first_BatchNormalization)
    #x = layers.Dropout(0.5)(x)
    x = layers.Dropout(0.5)(first_BatchNormalization)
    second_ConvLSTM = layers.ConvLSTM2D(filters=32, kernel_size=(2, 2)
                        , data_format='channels_last'
                        , activation='relu'
                        , padding='same', return_sequences=True)(x)

    second_BatchNormalization = layers.BatchNormalization()(second_ConvLSTM)
    
    x = layers.Dropout(0.5)(second_BatchNormalization)
    
    x = layers.Conv3D(
        filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same", strides=2
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    #x = layers.TimeDistributed(layers.Flatten())(x)


    x = layers.Flatten()(x)
    
    x = layers.Dense(units=1028, activation="relu",kernel_regularizer=tf.keras.regularizers.L1(0.01),
                                                   activity_regularizer=tf.keras.regularizers.L2(0.01))(x)
    #x = layers.Dense(units=1028, activation="relu")(flat_layer)
    #x = layers.Dropout(0.3)(x)

    
    #x = layers.Dropout(0.5)(x)

    #outputs = layers.Dense(units=1)(x)

    #model = keras.Model(inputs, outputs, name="convlstm")
    return keras.Model(inputs, x, name="convlstm")


## https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/

def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, 2, padding = 'same', strides = (1,2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, 2, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    #x = layers.Dropout(0.2)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv3D(filter, 1, strides = (1,2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, 2, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    #x = layers.Dropout(0.2)(x)
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, 2, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    #x = layers.Dropout(0.2)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet34_3d(width=128, height=128, depth=64, channels = 3):
    ## data_format="NDHWC"
    inputs3d = keras.Input((depth, width,height, channels))

    
    #x = tf.keras.layers.ZeroPadding3D(( 1,3, 3))(inputs3d)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv3D(64, kernel_size=2, strides=2, padding='same')(inputs3d)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2, strides=(1,2,2), padding='same')(x)
    x = layers.Dropout(0.4)(x)
    # Define size of sub-blocks and initial filter size
    #block_layers = [3, 4, 6, 3]
    block_layers = [3, 4,6,3]
    #filter_size = 64
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(len(block_layers)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
                
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    #x = layers.Dropout(0.4)(x)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling3D(3, padding = 'same',)(x)
    x = tf.keras.layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1028, activation='relu',
    kernel_regularizer=tf.keras.regularizers.L1(0.01),
    activity_regularizer=tf.keras.regularizers.L2(0.01))(x)
    #x = tf.keras.layers.Activation('relu')(x)
    #x = layers.Dense(units=1)(x)
    
    return tf.keras.models.Model(inputs = inputs3d, outputs = x, name = "ResNet34_3d")



def convolutional_blockvold(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, 2, padding = 'same', strides = (2,2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, 2, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    #x = layers.Dropout(0.2)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv3D(filter, 1, strides = (2,2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def identity_blockvold(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, 2, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    #x = layers.Dropout(0.2)(x)
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, 2, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    #x = layers.Dropout(0.2)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet34_3dvold(width=128, height=128, depth=64, channels = 3):
    ## data_format="NDHWC"
    inputs3d = keras.Input((depth, width, height, channels))


    
    x = tf.keras.layers.ZeroPadding3D(( 3,3, 3))(inputs3d)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv3D(64, kernel_size=2, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2, strides=2, padding='same')(x)
    x = layers.Dropout(0.4)(x)
    # Define size of sub-blocks and initial filter size
    #block_layers = [3, 4, 6, 3]
    block_layers = [3, 4,6,3]
    #filter_size = 64
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(len(block_layers)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
                
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    #x = layers.Dropout(0.4)(x)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling3D(3, padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(512, activation='relu',)(x)
    #x = tf.keras.layers.Activation('relu')(x)
    #x = layers.Dense(units=1)(x)
    
    return tf.keras.models.Model(inputs = inputs3d, outputs = x, name = "ResNet34_3d")



def set_mdensel_modelfirst(ncols = 6):

    inputs2d = keras.Input(shape = (ncols))

    x = layers.Dense(16, activation='relu')(inputs2d)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    #outputs = layers.Dense(units=1)(x)

    # Define the model.
    #model = keras.Model(inputs2d, outputs, name="morphomodel")
    return keras.models.Model(inputs2d,x,name='morphomodel')

def set_mdensel_model(ncols = 6):

    inputs2d = keras.Input(shape = (ncols))

    x = layers.Dense(16, activation='relu')(inputs2d)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    #outputs = layers.Dense(units=1)(x)

    # Define the model.
    #model = keras.Model(inputs2d, outputs, name="morphomodel")
    return keras.models.Model(inputs2d,x,name='morphomodel')


def set_Conv3dmodel_v2o(width=150, height=150, depth = 5, channels = 8):
    """Build a 3D convolutional neural network model."""
    #data_format="NDHWC"
    #inputs = keras.Input((width, height, depth, 3))
    inputs3d = keras.Input((depth,  width, height, channels))

    x = layers.Conv3D(filters=32, kernel_size=2, strides=2, padding='same', activation="relu")(inputs3d)
    x = layers.MaxPool3D(pool_size=2, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same',  activation="relu")(x)
    x = layers.MaxPool3D(pool_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same',  activation="relu")(x)
    x = layers.MaxPool3D(pool_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=32, kernel_size=2, strides=2, padding='same',  activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=1048, activation="relu")(x)

    return keras.Model(inputs3d, x, name="3dcnn")


def set_Conv3dmodel(width=150, height=150, depth = 5, channels = 8, initfilters = 32):
    """Build a 3D convolutional neural network model."""
    #data_format="NDHWC"
    #inputs = keras.Input((width, height, depth, 3))
    inputs3d = keras.Input((depth,  width, height, channels))

    x = layers.Conv3D(filters=initfilters, kernel_size=2, strides=2, padding='same', activation="relu")(inputs3d)
    x = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same',  activation="relu")(x)
    x = layers.MaxPool3D(pool_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same',  activation="relu")(x)
    x = layers.MaxPool3D(pool_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=32, kernel_size=2, strides=2, padding='same',  activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=1048, activation="relu")(x)

    return keras.Model(inputs3d, x, name="3dcnn")


def set_Conv3dmodel_vorig(width=150, height=150, depth = 5, channels = 8):
    """Build a 3D convolutional neural network model."""
    #data_format="NDHWC"
    #inputs = keras.Input((width, height, depth, 3))
    inputs3d = keras.Input((depth,  height, width,channels))

    x = layers.Conv3D(filters=32, kernel_size=2, strides=2, padding='same', activation="relu")(inputs3d)
    x = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, strides=2, padding='same',  activation="relu")(x)
    x = layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, strides=2, padding='same',  activation="relu")(x)
    x = layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(filters=32, kernel_size=2, strides=2, padding='same',  activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=1048, activation="relu")(x)



    #x = layers.Dense(units=512, activation="relu")(x)
    #x = layers.Dropout(0.3)(x)
    
    return keras.Model(inputs3d, x, name="3dcnn")

def final_model():
    input1 = set_mdensel_model()
    input2 = set_Conv3dmodel()

    # try to fix model_c here but i don't how 
    combined = keras.layers.concatenate([input2.output, input1.output],name="concatenate")
    combined = layers.Dropout(0.4)(combined)
    z = layers.Dense(units=128, activation="relu")(combined)
    z = layers.Dense(1)(z)
    model= keras.Model(inputs=[input2.input, input1.input], outputs=z, name = "final")

    return model

def alexnet3d(width=150, height=150, depth = 5, channels = 8):

    inputs3d = keras.Input((depth,  height, width,channels))

    x = keras.layers.Conv3D(filters=96, kernel_size=2, strides=2, padding='same', activation="relu")(inputs3d)
    x = keras.layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv3D(filters=256, kernel_size=5, strides=1, activation='relu', padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv3D(filters=384, kernel_size=3, strides=1, activation='relu', padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Conv3D(filters=384, kernel_size=3, strides=1, activation='relu', padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv3D(filters=256, kernel_size=3, strides=1, activation='relu', padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(1024, activation='relu')(x)


    return keras.Model(inputs3d, x, name="alexnet3d")
    
    


#https://medium.com/@AnasBrital98/inception-v4-cnn-architecture-explained-23a7fe12c727
def conv3d_with_Batch(prev_layer , nbr_kernels , filter_size , strides = 1 , padding = 'same'):
    x = keras.layers.Conv3D(filters = nbr_kernels, kernel_size = filter_size, strides=strides , padding=padding) (prev_layer)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation = 'relu') (x)
    return x

def stemBlock(prev_layer):
    x = conv3d_with_Batch(prev_layer, nbr_kernels = 32, filter_size = 3, strides = 2)
    x = conv3d_with_Batch(x, nbr_kernels = 32, filter_size = 3)
    x = conv3d_with_Batch(x, nbr_kernels = 64, filter_size = 3)
    
    x_1 = conv3d_with_Batch(x, nbr_kernels = 96, filter_size = 3, strides =2 )
    x_2 = keras.layers.MaxPool3D(pool_size=3 , strides=2 , padding='same') (x)
    
    x = keras.layers.concatenate([x_1 , x_2])
    
    x_1 = conv3d_with_Batch(x, nbr_kernels = 64, filter_size = 1)
    x_1 = conv3d_with_Batch(x_1, nbr_kernels = 64, filter_size = (1,7,7) , padding ='same')
    x_1 = conv3d_with_Batch(x_1, nbr_kernels = 64, filter_size = (7,7,1), padding ='same')
    x_1 = conv3d_with_Batch(x_1, nbr_kernels = 96, filter_size = 3)
    
    x_2 = conv3d_with_Batch(x, nbr_kernels = 96, filter_size = 1)
    x_2 = conv3d_with_Batch(x_2, nbr_kernels = 96, filter_size = 3)
    
    x = keras.layers.concatenate([x_1 , x_2])
    
    x_1 = conv3d_with_Batch(x, nbr_kernels = 192, filter_size = 3 , strides=2)
    x_2 = keras.layers.MaxPool3D(pool_size=3 , strides=2 , padding='same') (x)
    
    x = keras.layers.concatenate([x_1 , x_2])
    
    return x

def reduction_A_Block(prev_layer) :
    x_1 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = 1)
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 224, filter_size = 3 , padding='same')
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = 3 , strides=2) 
    
    x_2 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = 3 , strides=2)
    
    x_3 = keras.layers.MaxPool3D(pool_size=3 , strides=2 , padding='same')(prev_layer)
    
    x =  keras.layers.concatenate([x_1 , x_2 , x_3], axis = 3)
    
    return x

def reduction_B_Block(prev_layer):
    x_1 =  keras.layers.MaxPool3D(pool_size=3 , strides=2 , padding='same')(prev_layer)
    
    x_2 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = 1)
    x_2 = conv3d_with_Batch(prev_layer = x_2, nbr_kernels = 192, filter_size = 3 , strides=2 )
    
    x_3 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 256, filter_size = 1 )
    x_3 = conv3d_with_Batch(prev_layer = x_3, nbr_kernels = 256, filter_size = (1,7,7) , padding='same')
    x_3 = conv3d_with_Batch(prev_layer = x_3, nbr_kernels = 320, filter_size = (7,7,1) , padding='same')
    x_3 = conv3d_with_Batch(prev_layer = x_3, nbr_kernels = 320, filter_size = 3 , strides=2)
    
    x = keras.layers.concatenate([x_1 , x_2 , x_3], axis = 4)
    return x

def InceptionBlock_A(prev_layer): #I'm Here
    
    x_1 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 64, filter_size = 1)
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 96, filter_size = 3 , strides=1, padding='same' )
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 96, filter_size = 3 , strides=1 , padding='same')
    
    x_2 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 64, filter_size = 1)
    x_2 = conv3d_with_Batch(prev_layer = x_2, nbr_kernels = 96, filter_size = 3 , padding='same')
    
    x_3 = keras.layers.MaxPool3D(pool_size=3 , strides=1 , padding='same')(prev_layer)
    x_3 = conv3d_with_Batch(prev_layer = x_3, nbr_kernels = 96, filter_size = 1 , padding='same')
    
    x_4 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 96, filter_size = (1,1))
    
    output = keras.layers.concatenate([x_1 , x_2 , x_3 , x_4], axis = 3)

    return output
    
def InceptionBlock_B(prev_layer):
    
    x_1 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = 1)
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 192, filter_size = (7,7,1) , padding='same')
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 224, filter_size = (1,7,7) , padding='same')
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 224, filter_size = (7,7,1) , padding='same')
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (1,7,7), padding='same')
    
    x_2 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = 1)
    x_2 = conv3d_with_Batch(prev_layer = x_2, nbr_kernels = 224, filter_size = (1,7,7) , padding='same')
    x_2 = conv3d_with_Batch(prev_layer = x_2, nbr_kernels = 256, filter_size = (7,7,1), padding='same')
    
    x_3 = keras.layers.MaxPool3D(pool_size=3 , strides=1 , padding='same')(prev_layer)
    x_3 = conv3d_with_Batch(prev_layer = x_3, nbr_kernels = 128, filter_size = 1)
    
    x_4 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = 1)

    output = keras.layers.concatenate([x_1 , x_2 ,x_3, x_4], axis = 3) 
    return output


def InceptionBlock_C(prev_layer):
    
    x_1 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = 1)
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 448, filter_size = (3,3,1) , padding='same')
    x_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 512, filter_size = (1,3,3) , padding='same')
    x_1_1 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (1,3,3), padding='same')
    x_1_2 = conv3d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (3,3,1), padding='same')
    x_1 = keras.layers.concatenate([x_1_1 , x_1_2], axis = 3)
    
    x_2 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = 1)
    x_2_1 = conv3d_with_Batch(prev_layer = x_2, nbr_kernels = 256, filter_size = (1,3,3), padding='same')
    x_2_2 = conv3d_with_Batch(prev_layer = x_2, nbr_kernels = 256, filter_size = (3,3,1), padding='same')
    x_2 = keras.layers.concatenate([x_2_1 , x_2_2], axis = 3)
    
    x_3 = keras.layers.MaxPool3D(pool_size=3,strides = 1 , padding='same')(prev_layer)
    x_3 = conv3d_with_Batch(prev_layer = x_3, nbr_kernels = 256, filter_size = 3  , padding='same')
    
    x_4 = conv3d_with_Batch(prev_layer = prev_layer, nbr_kernels = 256, filter_size = 1)
    
    output = keras.layers.concatenate([x_1 , x_2 , x_3 , x_4], axis = 3)
    
    return output

def InceptionV4(width=150, height=150, depth = 5, channels = 8):

    input_layer = keras.Input((depth,  height, width,channels))
    
    x = stemBlock(prev_layer=input_layer)
    
    x = InceptionBlock_A(prev_layer=x)
    x = InceptionBlock_A(prev_layer=x)
    x = InceptionBlock_A(prev_layer=x)
    x = InceptionBlock_A(prev_layer=x)
    
    x = reduction_A_Block(prev_layer=x)
    
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    
    x = reduction_B_Block(prev_layer= x)
    
    x = InceptionBlock_C(prev_layer=x)
    x = InceptionBlock_C(prev_layer=x)
    x = InceptionBlock_C(prev_layer=x)
    
    x = layers.GlobalAveragePooling3D()(x)
    
    x = keras.layers.Dense(units = 1536, activation='relu') (x)
    x = keras.layers.Dropout(0.8)(x)
    #x = keras.layers.Dense(units = 1000, activation='softmax')(x)
    
    model = keras.Model(inputs = input_layer , outputs = x , name ='Inception-V4')
    
    return model