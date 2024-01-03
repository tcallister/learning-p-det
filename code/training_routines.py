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

        """
        Parameters
        ----------
        y_true : `list`
            True missed/found labels (0/1 respectively)
        y_pred : `list`
            Corresponding set of predicted detection probabilities
        """

        # The log likelihood below diverges numerically if predicted probabilities are too close to unity
        # (note that this is a numerical precision issue, not anything fundamental). Accordingly, implement
        # a ceiling value of P_det = 1-1e-9 to ensure that the loss function remains finite
        ceil = tf.ones_like(y_pred)*(1.-1e-9)
        y_pred = tf.where(y_pred>1.-1e-9,ceil,y_pred)

        floor = tf.ones_like(y_pred)*(1e-20)
        y_pred = tf.where(y_pred<1e-20,floor,y_pred)

        # Binomial log likelihood (aka cross-entropy loss fucntion)
        log_ps = tf.where(y_true==1,tf.math.log(y_pred),tf.math.log(1.-y_pred))
        return -tf.math.reduce_mean(log_ps)

def scheduler(epoch, lr):

    if epoch % 3 == 0 and epoch:
        return lr*tf.math.exp(-0.5)
    else:
        return lr

def build_ann(input_shape=9,layer_width=64,hidden_layers=3,lr=1e-3,leaky_alpha=0.01,output_bias=None):

    """
    Function to construct and return an ANN object, to be subsequently trained or into which
    pre-trained weights can be loaded.

    Parameters
    ----------
    input_shape : `int`
        Dimensionality of input feature space (default 9)
    layer_width : `int`
        Number of neurons in each hidden layer (default 64)
    hidden_layers : `int`
        Number of hidden layers (default 3)
    lr : `float`
        Learning rate (default 1e-3)
    leaky_alpha : `float` 
        Parameter specifying LeakyReLU activation function (default 0.01)

    Returns
    -------
    ann : `tf.keras.model.Sequential()`
        Compiled ANN object
    """

    # Initialize a sequential ANN object and create an initial hidden layer
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=layer_width, input_shape=(input_shape,),
                                  kernel_initializer=initializers.RandomUniform(),
                                  bias_initializer=initializers.RandomNormal(stddev=0.01)))
            
    # Activation function
    ann.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))

    # Add the specified number of additional hidden layers, each with another activation
    for i in range(hidden_layers-1):
        ann.add(tf.keras.layers.Dense(units=layer_width))
                                      #kernel_initializer=initializers.RandomUniform(),
                                      #bias_initializer=initializers.RandomNormal(stddev=0.01)))
        #ann.add(tf.keras.layers.Dropout(0.5))
        ann.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))

    # Final output layer with sigmoid activation
    ann.add(tf.keras.layers.Dropout(0.5))
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    ann.add(tf.keras.layers.Dense(units=1,bias_initializer=output_bias,activation='sigmoid'))
    
    # Other setup
    loss = negativeLogLikelihood()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    ann.compile(optimizer = opt,
                loss = loss,
                metrics = ['accuracy',tf.keras.metrics.Precision(name='precision')])
    
    return ann

