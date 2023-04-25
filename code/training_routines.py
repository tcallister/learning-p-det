import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras import initializers

class negativeLogLikelihood(tf.keras.losses.Loss):

    """
    Custom loss function implementing binomial detection likelihood model
    """

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):

        ceil = tf.ones_like(y_pred)*(1.-1e-9)
        y_pred = tf.where(y_pred>1.-1e-9,ceil,y_pred)

        log_ps = tf.where(y_true==1,tf.math.log(y_pred),tf.math.log(1.-y_pred))
        return -tf.math.reduce_mean(log_ps)

def scheduler(epoch, lr):

    if epoch % 3 == 0 and epoch:
        return lr*tf.math.exp(-0.5)
    else:
        return lr

def build_ann(input_shape=9,layer_width=64,hidden_layers=3,lr=1e-3,leaky_alpha=0.01):

    # Initialize a sequential ANN object and create an initial hidden layer
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=layer_width, input_shape=(input_shape,),
                                  kernel_initializer=initializers.RandomUniform(),
                                  bias_initializer=initializers.RandomNormal(stddev=0.01)))
            
    # Activation function
    ann.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))

    # Add the specified number of additional hidden layers, each with another activation
    for i in range(hidden_layers-1):
        ann.add(tf.keras.layers.Dense(units=layer_width,
                                      kernel_initializer=initializers.RandomUniform(),
                                      bias_initializer=initializers.RandomNormal(stddev=0.01)))
        ann.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    
    # Final output layer with sigmoid activation
    ann.add(tf.keras.layers.Dense(units=1))
    ann.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
    
    # Other setup
    loss = negativeLogLikelihood()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    ann.compile(optimizer = opt,
                loss = loss,
                metrics = ['accuracy'])
    
    return ann

