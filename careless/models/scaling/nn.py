import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
from careless.models.scaling.base import Scaler
import numpy as np



class ResNetLayer(tfk.layers.Layer):
    def __init__(
            self,
            activation='ReLU', 
            kernel_initializer=None,
            dropout=None,
        ):
        super().__init__()
        self.kernel_initializer = kernel_initializer
        self.activation = tfk.activations.get(activation)
        if dropout is None:
            self.dropout = None
        else:
            self.dropout = tfk.layers.Dropout(dropout)

    def build(self, shapes, **kwargs):
        width = shapes[-1]

        if self.kernel_initializer is None:
            kernel_initializer = tfk.initializers.variance_scaling(0.1, mode='fan_avg')
        else:
            kernel_initializer = self.kernel_initializer
        
        self.dense_1 = tfk.layers.Dense(
            width,
            kernel_initializer=kernel_initializer,
        )

        if self.kernel_initializer is None:
            kernel_initializer = tfk.initializers.variance_scaling(0.1, mode='fan_avg')
        else:
            kernel_initializer = self.kernel_initializer

        self.dense_2 = tfk.layers.Dense(
            width,
            kernel_initializer=kernel_initializer,
        )

    def call(self, x, **kwargs):
        out = x
        out = self.activation(out)
        out = self.dense_1(out)
        out = self.activation(out)
        out = self.dense_2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out + x
        return out

class NormalLayer(tf.keras.layers.Layer):
    def __init__(self, scale_bijector=None, epsilon=1e-7, **kwargs): 
        super().__init__(**kwargs)
        self.epsilon = epsilon
        if scale_bijector is None:
            self.scale_bijector = tfb.Chain([
                tfb.Shift(epsilon),
                tfb.Exp(),
            ])
        else:
            self.scale_bijector = scale_bijector

    def call(self, x, **kwargs):
        loc, scale = tf.unstack(x, axis=-1)
        scale = self.scale_bijector(scale)
        return tfd.Normal(loc, scale)

class MetadataScaler(Scaler):
    """
    Neural network based scaler with simple dense layers.
    This neural network outputs a normal distribution.
    """
    def __init__(self, n_layers, width, leakiness=0.01, epsilon=1e-5, dropout=0.1):
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

        kernel_initializer = tfk.initializers.variance_scaling(0.1, mode='fan_avg')
        mlp_layers = [
            tfk.layers.Dense(width, kernel_initializer=kernel_initializer)
        ]

        for i in range(n_layers):
            mlp_layers.append(ResNetLayer(
                    activation='ReLU', 
                    kernel_initializer=None,
                    dropout=dropout,
                    )
                )
        mlp_layers.append(tfk.layers.LayerNormalization())

        #The last layer is linear and generates location/scale params
        tfp_layers = []
        tfp_layers.append(
            tf.keras.layers.Dense(
                2, 
                activation='linear', 
                use_bias=True, 
                kernel_initializer='glorot_normal',
            )
        )

        #The final layer converts the output to a Normal distribution
        #tfp_layers.append(tfp.layers.IndependentNormal())
        tfp_layers.append(NormalLayer(epsilon=epsilon))

        self.network = tf.keras.Sequential(mlp_layers)
        self.distribution = tf.keras.Sequential(tfp_layers)

    def call(self, metadata):
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
    def call(self, inputs):
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

