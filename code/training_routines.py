import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras import initializers
from sklearn.preprocessing import StandardScaler
from utilities import load_training_data
from draw_new_injections import draw_new_injections
import numpy as np
from tqdm import tqdm
import sys

class NegativeLogLikelihood(tf.keras.losses.Loss):

    """
    Custom loss function implementing binomial detection likelihood model, with
    a prior penalizing large predicted detection probabilities.
    """

    def __init__(self, beta=2./3.):

        """
        Initialize an instance of the loss function.

        Parameters
        ----------
        beta : `float`
            Parameter specifying the degree of penalty applied to large
            predicted detection probabilities. Larger values correspond to
            steeper penalty. Default is `2/3`

        Returns
        -------
        Class instance
        """

        self.beta = beta
        super().__init__()

    def call(self, y_true, y_pred):

        """
        Function to evaluate loss, given true found/missed labels and
        predicted detection probabilities.

        Parameters
        ----------
        y_true : `list`
            True missed/found labels (0/1 respectively)
        y_pred : `list`
            Corresponding set of predicted detection probabilities

        Returns
        -------
        loss : `tensorflow.Tensor`
            Negative log likelihood of given predictions.
        """

        # Binomial log likelihood (aka cross-entropy loss fucntion)
        log_ps = tf.where(y_true == 1,
                          tf.math.log(y_pred),
                          tf.math.log(1.-y_pred))

        # Return with prior penalizing large probabilities
        return -tf.math.reduce_mean(log_ps) \
            + tf.math.reduce_mean(self.beta*y_pred)


def NegativeLogLikelihoodAugmented(y_true, y_pred, beta,
            efficiency_mismatches=tf.convert_to_tensor([0.], dtype='float64')):

    """
    Custom loss function implementing binomial likelihood model and penalizing
    large detection probabilities. Further includes terms grading the network
    on its integrated detection efficiencies predicted for an arbitrary number
    of reference populations.

    Parameters
    ----------
    y_true : `list`
        True missed/found labels (0/1 respectively)
    y_pred : `list`
        Corresponding set of predicted detection probabilities
    beta : `float`
        Inverse temperature penalty on large predicted detection probabilities
    efficiency_mismatches : `tf.Tensor`
        Standardized residuals between expected and predicted detection
        efficiences for an arbitrary number of reference populations. These
        should be calculated as `(f_pred - f_true)/std`, where `f_pred` is
        the predicted efficiency using the network, `f_true` is the target
        efficiency, and the expected root-variance `std` is calculable using
        the number `N` of events used to compute `f_pred`. Default `[0]`.

    Returns
    -------
    loss : `tensorflow.Tensor`
        Negative log likelihood of given predictions.
    """

    # Binomial log likelihood (aka cross-entropy loss function)
    log_ps = tf.where(y_true == 1,
                      tf.math.log(y_pred),
                      tf.math.log(1.-y_pred))
    term1 = -tf.math.reduce_mean(log_ps)

    # Negative log likelihood of predicted detection efficiencies
    # Provided values should be pre-standardized:
    # (Predicted-Actual)**2/(Expected variance)
    term2 = tf.math.reduce_sum(efficiency_mismatches/2.)

    # Return with prior penalizing large probabilities
    term3 = tf.math.reduce_mean(beta*y_pred)

    return term1+term2+term3


def scheduler(epoch, lr):

    """
    Example scheduler to reduce learning rate over the course of network
    training.

    Parameters
    ----------
    epoch : `int`
        Training epoch
    lr : `float`
        Current learning rate

    Returns
    -------
    lr : `float`
        New learning rate
    """

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
              kernel_init='Glorot',
              bias_init='Zeros',
              dropout=True,
              dropout_rate=0.5,
              output_bias=None,
              output_activation='sigmoid'):

    """
    Function to construct and return an ANN object, to be subsequently trained
    or into which pre-trained weights can be loaded.

    Parameters
    ----------
    input_shape : `int`
        Dimensionality of input feature space (default 9)
    layer_width : `int`
        Number of neurons in each hidden layer (default 64)
    hidden_layers : `int`
        Number of hidden layers (default 3)
    loss : `func` or `None`
        Loss function for use in training. If `None`, network will default to
        using `NegativeLogLikelihood`. Argument defaults to `None`.
    lr : `float`
        Learning rate (default 1e-3)
    activation : `str`
        String specifying activation functions to be applied to initial and
        hidden layers. One of `ReLU`, `LeakyReLU`, `ELU`, `sigmoid`, or
        `swish`. Defaults to `ReLU`.
    leaky_alpha : `float`
        Parameter specifying LeakyReLU activation function (default 0.01).
        Used only when `activation == "LeakyReLU"`.
    kernel_init : `str` or `tf.keras.Initializer`
        Specifies initialization strategy for kernel weights. Set
        `kernel_init="Glorot"` to use `tf.keras.initializers.GlorotUniform`.
        Otherwise, a `tf.keras.Initializer` can be directly passed. Defaults
        to `Glorot`.
    bias_init : `str` or `tf.keras.Initializer`
        Specifies initialization strategy for neuron biases. Set
        `kernel_init="Zeros"` to use `tf.keras.initializers.Zeros`.
        Otherwise, a `tf.keras.Initializer` can be directly passed. Defaults
        to `Zeros`.
    dropout : `bool`
        Determines whether a dropout layer is included preceding the final
        output layer. Default is `True`
    dropout_rate : `float`
        Dropout rate to apply to the dropout layer. Default is `0.5`
    output_bias : `float` or `None`
        Bias to include in output layer. If `None`, no bias is included.
        Default is `None`.
    output_activation : `str` or `func`
        Activation function to apply to the output layer. One of `sigmoid`,
        `negative_exponential`, or `scaled_sigmoid`, or a callable function.
        Default is `sigmoid`.

    Returns
    -------
    ann : `tf.keras.model.Sequential()`
        Compiled ANN object
    """

    # Set kernel initializer, if passed as a string
    if kernel_init == 'Glorot':
        kernel_init = initializers.GlorotUniform()

    # Set bias initializer, if passed as a string
    if bias_init == 'Zeros':
        bias_init = initializers.Zeros()

    # Activation function
    if activation == 'ReLU':
        act = tf.keras.layers.ReLU()
    elif activation == 'LeakyReLU':
        act = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)
    elif activation == 'ELU':
        act = tf.keras.layers.ELU()
    elif activation == 'sigmoid':
        act = tf.keras.layers.Activation('sigmoid')
    elif activation == 'swish':
        act = tf.keras.layers.Activation('swish')
    else:
        print("Activation not recognized!")
        sys.exit()

    # Initialize a sequential ANN object and create an initial hidden layer
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=layer_width, input_shape=(input_shape,),
                                  kernel_initializer=kernel_init,
                                  bias_initializer=bias_init))

    # Activation function
    ann.add(act)

    # Add the specified number of additional hidden layers, each with another
    # activation
    for i in range(hidden_layers-1):

        # Dense layer
        ann.add(tf.keras.layers.Dense(units=layer_width,
                                      kernel_initializer=kernel_init,
                                      bias_initializer=bias_init))

        # Activation
        ann.add(act)

    # Add dropout, if specified
    if dropout:
        print("!!", dropout_rate)
        ann.add(tf.keras.layers.Dropout(dropout_rate))

    # Final output layer
    # First set bias, as specified
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Set output activation function
    if output_activation == 'negative_exponential':
        def neg_exp_activation(x):
            return -tf.exp(x)
        output_activation = neg_exp_activation

    elif output_activation == 'scaled_sigmoid':
        def scaled_sigmoid(x):
            return (1.-0.0589) * tf.nn.sigmoid(x)
        output_activation = scaled_sigmoid

    # Add output layer
    ann.add(tf.keras.layers.Dense(units=1,
                                  bias_initializer=output_bias,
                                  activation=output_activation))

    # Other setup
    if not loss:
        loss = NegativeLogLikelihood()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile and return
    ann.compile(optimizer=opt,
                loss=loss,
                metrics=['accuracy',
                         tf.keras.metrics.Precision(name='precision')])

    return ann


class NeuralNetworkWrapper:

    """
    Wrapper class used to create and train a neural network Pdet emulator.
    Used instead of `build_ann` in order to use a more complex loss function
    (e.g. `NegativeLogLikelihoodAugmented`) that requires use of a manual
    training loop, rather than more automated tensorflow training tools.
    """

    def __init__(self,
                 input_shape=9,
                 layer_width=64,
                 hidden_layers=3,
                 loss=None,
                 lr=1e-3,
                 activation='ReLU',
                 leaky_alpha=0.01,
                 output_bias=0,
                 addDerived=lambda x: None,
                 feature_names=[]):

        """
        Instantiates and returns a `NeuralNetworkWrapper` object, containing
        a prepared neural network.

        Parameters
        ----------
        input_shape : `int`
            Dimensionality of input feature space (default 9)
        layer_width : `int`
            Number of neurons in each hidden layer (default 64)
        hidden_layers : `int`
            Number of hidden layers (default 3)
        loss : `func` or `None`
            Loss function for use in training. If `None`, defaults to
            `NegativeLogLikelihood`. Default is `None`.
        lr : `float`
            Learning rate (default 1e-3)
        activation : `str`
            One of `ReLU`, `LeakyReLU`, or `ELU`. Default `ReLU`.
        leaky_alpha : `float`
            Parameter specifying LeakyReLU activation function (default 0.01)
        output_bias : `float`
            Bias to include in output layer (default 0)
        addDerived : `func`
            Function, if needed, to add derived data columns to input data.
            Defaults to the identity, such that no derived data is added.
        feature_names : `list`
            Parameters to extract from data for use in neural network.

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
        self.output_bias = output_bias

        # Instantiate neural network
        self.model = self.build_model()

        # Store specified function for data augmentation and list of
        # features to extract
        self.addDerived = addDerived
        self.feature_names = feature_names

        # Prepare attributes for storing training/testing data
        self.input_scaler = None
        self.train_data = None
        self.test_data = None

        # List to hold auxiliary datasets used to incorporate
        # predicted integrated detection efficiencies during training
        self.auxiliary_data = []

        # Training history
        self.loss_history = []
        self.val_loss_history = []

    def build_model(self):

        """
        Function to construct and return an ANN object, to be subsequently
        trained or into which pre-trained weights can be loaded.

        Parameters
        ----------
        None

        Returns
        -------
        ann : `tf.keras.model.Sequential()`
            Compiled ANN object
        """

        # Set up chosen properties
        if self.activation == 'ReLU':
            activation = tf.keras.layers.ReLU()
        elif self.activation == 'LeakyReLU':
            activation = tf.keras.layers.LeakyReLU(alpha=self.leaky_alpha)
        elif self.activation == 'ELU':
            activation = tf.keras.layers.ELU()
        else:
            print("Activation not recognized!")
            sys.exit()

        # Initialize a sequential ANN object and create an initial hidden layer
        ann = tf.keras.models.Sequential()
        ann.add(
            tf.keras.layers.Dense(
                units=self.layer_width,
                input_shape=(self.input_shape,),
                kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.01),
                bias_initializer=initializers.Zeros()))

        # Add activation function
        ann.add(activation)

        # Add the specified number of additional hidden layers, each with
        # another activation
        for i in range(self.hidden_layers-1):

            ann.add(
                tf.keras.layers.Dense(
                    units=self.layer_width,
                    kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.01),
                    bias_initializer=initializers.Zeros()))

            ann.add(activation)

        # Prepare output bias
        output_bias = tf.keras.initializers.Constant(self.output_bias)

        # Final output layer with sigmoid activation
        # This provides a hard cap on predicted probabilities, accounting for
        # time in which no interferometers were on
        def scaled_sigmoid(x):
            return (1.-0.0589) * tf.nn.sigmoid(x)
        ann.add(tf.keras.layers.Dense(units=1,
                                      bias_initializer=output_bias,
                                      activation=scaled_sigmoid))

        return ann

    def prepare_data(self,
                     batch_size,
                     train_data_external,
                     val_data_external):
        """
        Prepare the training and validation data to be used during training.
        Accepts training and validation datasets, and uses `self.addDerived`
        and `self.feature_names`, provided at the time of class creation,
        to augment the provided data, extract relevant features, split into
        input and output columns, and rescale inputs.

        Parameters
        ----------
        batch_size : `int`
            Specifies batch size to be used during network training
        train_data_external : `numpy.ndarray`
            Array of data for neural network during training
        val_data_external : `numpy.ndarray`
            Array of validation data for neural network during testing

        Returns
        -------
        None
        """

        # Make copy of data for safety
        train_data = train_data_external.copy()
        val_data = val_data_external.copy()

        # Add derived parameters
        self.addDerived(train_data)
        self.addDerived(val_data)

        # Split off inputs and outputs
        train_input = train_data[self.feature_names].values
        train_output = train_data['detected'][:, np.newaxis]
        val_input = val_data[self.feature_names].values
        val_output = val_data['detected'][:, np.newaxis]

        # Define quantile transformer and scale inputs
        # Store input scaler
        self.input_scaler = StandardScaler()
        self.input_scaler.fit(train_input)
        train_input_scaled = self.input_scaler.transform(train_input)
        val_input_scaled = self.input_scaler.transform(val_input)

        # Create a tf.data.Dataset for the training data
        train_dataset = tf.data.Dataset.from_tensor_slices((train_input_scaled,
                                                            train_output))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
        train_dataset = train_dataset.batch(batch_size)

        # Create a tf.data.Dataset for the validation data
        val_dataset = tf.data.Dataset.from_tensor_slices((val_input_scaled,
                                                          val_output))
        val_dataset = val_dataset.batch(batch_size)

        # Save as attributes
        self.train_data = train_dataset
        self.test_data = val_dataset

        return

    def draw_from_reference_population(self,
                                       parameter_dict,
                                       n_draws,
                                       target_efficiency):

        """
        Function to draw from a reference population of synthetic data, to be
        used for auxiliary training data. Specifically, the loss function used
        during training will be further penalized based on the network's
        predicted detection efficiencies integrated over this reference
        population, compared to the expected truth.

        Parameters
        ----------
        parameter_dict : `dict`
            Dictionary of population parameters to be used for drawing random
            compact binaries
        n_draws : `int`
            Number of synthetic samples to draw
        target_efficiency : `float`
            Target recovery efficiency for the synthetic samples

        Returns
        -------
        None
        """

        # Draw new samples, extract training features, and transform
        new_draws = draw_new_injections(batch_size=n_draws, **parameter_dict)
        self.addDerived(new_draws)
        new_draws = self.input_scaler.transform(new_draws[self.feature_names])

        # Expected standard deviation in recovered draws
        std = np.sqrt(target_efficiency/n_draws)
        print("Expected std:", std)

        # Save draws and target recovery efficiency to class' auxiliary data
        self.auxiliary_data.append((tf.convert_to_tensor(new_draws),
                                    target_efficiency,
                                    std))

    def train_model(self, epochs, beta):

        """
        Class method that implements network training.

        Parameters
        ----------
        epochs : `int`
            Number of training epochs
        beta : `float`
            Parameter penalizing large predicted detection probabilities.
        """

        # Define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Define early stopping parameters
        best_val_loss = float('inf')
        best_epoch = 0
        best_weights = None
        wait = 0
        patience = 10

        # Loop across training epochs
        for epoch in range(epochs):

            # Loop across epochs 
            epoch_losses = []
            for step, (x_batch_train, y_batch_train) in tqdm(enumerate(self.train_data), total=len(self.train_data)):

                #loss_value = train_step(x_batch_train,y_batch_train)
                #epoch_losses.append(loss_value)

                # Prepare gradient
                with tf.GradientTape() as tape:

                    # Compute predicted detection probabilities on training data
                    y_pred_train = (self.model(x_batch_train, training=True))

                    # Compute predicted detection efficiencies on preloaded populations
                    efficiencies = tf.transpose([tf.reduce_mean(self.model(auxiliary_data[0], training=True)) for auxiliary_data in self.auxiliary_data])

                    # Compute efficiency mismatches
                    target_efficiencies = tf.convert_to_tensor([auxiliary_data[1] for auxiliary_data in self.auxiliary_data],dtype='float64')
                    std_efficiencies = tf.convert_to_tensor([auxiliary_data[2] for auxiliary_data in self.auxiliary_data],dtype='float64')
                    efficiency_mismatch = (efficiencies-target_efficiencies)**2/std_efficiencies**2

                    # Compute the loss using both the training predictions and the efficiency predictions
                    loss_value = self.loss(y_batch_train, y_pred_train, beta, efficiency_mismatch)

                # Compute gradient, update weights, and save loss
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                epoch_losses.append(loss_value)
                
            # Compute mean loss and validation loss at the end of the epoch
            loss = np.mean(epoch_losses)
            val_loss = np.mean([self.loss(y, self.model(x, training=False), beta) for x, y in self.test_data])
            #val_loss = test.step(x,y)
            self.loss_history.append(loss)
            self.val_loss_history.append(val_loss)
            print("Epoch: {}, Loss: {}, Val Loss: {}".format(epoch, loss, val_loss))

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
