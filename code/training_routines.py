import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras import initializers
from utilities import load_training_data
import numpy as np
from tqdm import tqdm

class NegativeLogLikelihood(tf.keras.losses.Loss):

    """
    Custom loss function implementing binomial detection likelihood model
    """

    def __init__(self,beta=2./3.):
        self.beta = beta
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
        #ceil = tf.ones_like(y_pred)*(1.-1e-20)
        #y_pred = tf.where(y_pred>1.-1e-20,ceil,y_pred)

        #floor = tf.ones_like(y_pred)*(1e-40)
        #y_pred = tf.where(y_pred<1e-40,floor,y_pred)

        # Binomial log likelihood (aka cross-entropy loss fucntion)
        log_ps = tf.where(y_true==1,tf.math.log(y_pred),tf.math.log(1.-y_pred))

        # Return with prior penalizing large probabilities
        return -tf.math.reduce_mean(log_ps) + tf.math.reduce_mean(self.beta*y_pred)

def scheduler(epoch, lr):

    if epoch % 3 == 0 and epoch:
        return lr*tf.math.exp(-0.5)
    else:
        return lr

def build_ann(input_shape=9,
        layer_width=64,
        hidden_layers=3,
        loss=None,
        lr=1e-3,
        activation='ReLU',
        leaky_alpha=0.01,
        dropout=True,
        dropout_rate=0.5,
        output_bias=None):

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
                                  kernel_initializer=initializers.GlorotUniform(),
                                  bias_initializer=initializers.Zeros()))
            
    # Activation function
    if activation=='ReLU':
        ann.add(tf.keras.layers.ReLU())
    elif activation=='LeakyReLU':
        ann.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    elif activation=='ELU':
        ann.add(tf.keras.layers.ELU())
    else:
        print("Activation not recognized!")
        sys.exit()

    # Add the specified number of additional hidden layers, each with another activation
    for i in range(hidden_layers-1):

        # Dense layer
        ann.add(tf.keras.layers.Dense(units=layer_width,
                                  kernel_initializer=initializers.GlorotUniform(),
                                  bias_initializer=initializers.Zeros()))

        # Activation
        if activation=='ReLU':
            ann.add(tf.keras.layers.ReLU())
        elif activation=='LeakyReLU':
            ann.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
        elif activation=='ELU':
            ann.add(tf.keras.layers.ELU())


    # Add dropout, if specified
    if dropout:
        print("!!",dropout_rate)
        ann.add(tf.keras.layers.Dropout(dropout_rate))

    # Final output layer with sigmoid activation
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    ann.add(tf.keras.layers.Dense(units=1,bias_initializer=output_bias,activation='sigmoid'))
    
    # Other setup
    if not loss:
        loss = negativeLogLikelihood()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    ann.compile(optimizer = opt,
                loss = loss,
                metrics = ['accuracy',tf.keras.metrics.Precision(name='precision')])
    
    return ann

class NeuralNetworkWrapper:

    def __init__(self,
                 input_shape=9,
                 layer_width=64,
                 hidden_layers=3,
                 loss=None,
                 lr=1e-3,
                 activation='ReLU',
                 leaky_alpha=0.01,
                 dropout=False,
                 dropout_rate=0.5,
                 output_bias=0):
        
        """
        Class to store parameters for a neural network and instantiate a model object

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
        dropout : `bool`
            Flag to include dropout layer (default True)
        dropout_rate : `float`
            Dropout rate (default 0.5)
        output_bias : `float`
            Bias to include in output layer (default None)
        
        Returns
        -------
        None
        """
        
        # Store parameters
        self.input_shape = input_shape
        self.layer_width = layer_width
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.lr = lr
        self.activation = activation
        self.leaky_alpha = leaky_alpha
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.output_bias = output_bias

        # Instantiate neural network
        self.model = self.build_model()

        # Initialize training and testing data
        self.train_data = None
        self.test_data = None
        self.auxiliary_data = None

        # Training history
        self.loss_history = []
        self.val_loss_history = []

    def build_model(self):

        """
        Function to construct and return an ANN object, to be subsequently trained or into which
        pre-trained weights can be loaded.
        
        Returns
        -------
        ann : `tf.keras.model.Sequential()`
            Compiled ANN object
        """

        # Initialize a sequential ANN object and create an initial hidden layer
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=self.layer_width,
                                      input_shape=(self.input_shape,),
                                      kernel_initializer=initializers.GlorotUniform(),
                                      bias_initializer=initializers.Zeros()))
        
        # Activation function
        if self.activation == 'ReLU':
            ann.add(tf.keras.layers.ReLU())
        elif self.activation == 'LeakyReLU':
            ann.add(tf.keras.layers.LeakyReLU(alpha=self.leaky_alpha))
        elif self.activation == 'ELU':
            ann.add(tf.keras.layers.ELU())
        else:
            print("Activation not recognized!")
            sys.exit()
        
        # Add the specified number of additional hidden layers, each with another activation
        for i in range(self.hidden_layers-1):
            ann.add(tf.keras.layers.Dense(units=self.layer_width,
                                            kernel_initializer=initializers.GlorotUniform(),
                                            bias_initializer=initializers.Zeros()))
            
            if self.activation == 'ReLU':
                ann.add(tf.keras.layers.ReLU())
            elif self.activation == 'LeakyReLU':
                ann.add(tf.keras.layers.LeakyReLU(alpha=self.leaky_alpha))
            elif self.activation == 'ELU':
                ann.add(tf.keras.layers.ELU())
        
        # Add dropout, if specified
        #if self.dropout:
        #    ann.add(tf.keras.layers.Dropout(self.dropout_rate))
        
        # Add output bias, if specified
        if self.output_bias is not None:
            output_bias = tf.keras.initializers.Constant(self.output_bias)
        
        # Final output layer with sigmoid activation
        ann.add(tf.keras.layers.Dense(units=1, bias_initializer=output_bias, activation='sigmoid'))
        
        return ann
    
    def prepare_data(self,
                    batch_size,
                    train_data_input,
                    train_data_output,
                    test_data_input,
                    test_data_output):

        print(train_data_input.shape,train_data_output.shape)
        
        # Create a tf.data.Dataset for the training data
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data_input,train_data_output))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset)).batch(batch_size)

        # Create a tf.data.Dataset for the validation data
        val_dataset = tf.data.Dataset.from_tensor_slices((test_data_input,test_data_output))
        val_dataset = val_dataset.batch(batch_size)

        self.train_data = train_dataset
        self.test_data = val_dataset
    
    def train_model(self, epochs):

        # Define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Define early stopping parameters
        best_val_loss = float('inf')
        best_epoch = 0
        best_weights = None
        wait = 0
        patience = 10

        def train_step(x,y):

            with tf.GradientTape() as tape:

                # Run the model on the training data
                y_pred_train = (self.model(x_batch_train, training=True))

                # Compute the loss using both the training predictions and the external predictions
                loss_value = self.loss(y_batch_train, y_pred_train)#, y_pred_external)

            grads = tape.gradient(loss_value, self.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            return loss_value

        # Custom training loop
        for epoch in range(epochs):

            epoch_losses = []

            for step, (x_batch_train, y_batch_train) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                loss_value = train_step(x_batch_train, y_batch_train)
                epoch_losses.append(loss_value)
                
            # Compute validation loss at the end of the epoch
            loss = np.mean(epoch_losses)
            val_loss = np.mean([self.loss(y, self.model(x, training=False)) for x, y in self.test_data])
            print("Epoch: {}, Loss: {}, Val Loss: {}".format(epoch, loss, val_loss))

            self.loss_history.append(loss)
            self.val_loss_history.append(val_loss)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping, reverting to best epoch: {}".format(best_epoch))
                    self.model.set_weights(best_weights)
                    break  # Early stopping condition met
