import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
from careless.models.scaling.base import Scaler
from tensorflow import keras as tfk
import numpy as np




class ResnetLayer(tfk.layers.Layer):
    def __init__(self, units, activation='ReLU', **kwargs):
        super().__init__()
        self._dense_kwargs = kwargs
        self.activation_1 = tfk.activations.get(activation)
        self.activation_2 = tfk.activations.get(activation)
        self.units = units

    def build(self, shape, **kwargs):
        self.dense_1 = tfk.layers.Dense(self.units, **self._dense_kwargs)
        self.dense_2 = tfk.layers.Dense(shape[-1], **self._dense_kwargs)

    def call(self, X, **kwargs):
        out = X
        out = self.activation_1(out)
        out = self.dense_1(out)
        out = self.activation_2(out)
        out = self.dense_2(out)
        return out + X


class MetadataScaler(Scaler):
    """
    Neural network based scaler with simple dense layers.
    This neural network outputs a normal distribution.
    """
    def __init__(self, n_layers, width, leakiness=0.01):
        """
        Parameters
        ----------
        n_layers : int 
            Number of layers
        width : int
            Width of layers
        leakiness : float or None
            If float, use LeakyReLU activation with provided parameter. Otherwise 
            use a simple ReLU
        """
        super().__init__()

        mlp_layers = []

        kernel_initializer = tfk.initializers.VarianceScaling(
            scale = 1./5./n_layers,
            mode='fan_avg', 
            distribution='truncated_normal',
        )

        for i in range(n_layers):
            mlp_layers.append(
                ResnetLayer(2*width, kernel_initializer=kernel_initializer)
            )

        #The last layer is linear and generates location/scale params
        tfp_layers = []
        tfp_layers.append(
            tf.keras.layers.Dense(
                tfp.layers.IndependentNormal.params_size(), 
                activation='linear', 
                use_bias=True, 
                kernel_initializer=kernel_initializer
            )
        )

        #The final layer converts the output to a Normal distribution
        tfp_layers.append(tfp.layers.IndependentNormal())

        self.network = tf.keras.Sequential(mlp_layers)
        self.distribution = tf.keras.Sequential(tfp_layers)

    def call(self, metadata, **kwargs):
        """
        Parameters
        ----------
        metadata : tf.Tensor(float32)

        Returns
        -------
        dist : tfp.distributions.Distribution
            A tfp distribution instance.
        """
        return self.distribution(self.network(metadata))


class MLPScaler(MetadataScaler):
    def call(self, inputs, **kwargs):
        """
        Parameters
        ----------
        inputs : tf.Tensor(float32)
            An arbitrarily batched input tensor

        Returns
        -------
        dist : tfp.distributions.Distribution
            A tfp distribution instance.
        """
        metadata = self.get_metadata(inputs)
        return super().call(metadata)

