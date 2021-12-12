from torch import nn
from neural_network.base_network import BaseNetwork, get_nonlinearity


class MlpNetwork(BaseNetwork):
    def __init__(
                self,
                input_size,
                output_size,
                hidden_layer=[128, 128],
                activation="relu",
                output_activation=None,
                network_name="mlp"):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_list = hidden_layer
        self.activation_name = activation
        self.output_activation_name = "identity" if output_activation is None else output_activation
        self.network_name = network_name

        self._get_activation()
        self._get_linear_layer()
        self.layer = [(linear, activation) for linear, activation in zip(self.linear_layer, self.activation)]

    def _get_activation(self):
        hidden_length = len(self.hidden_layer_list)
        self.activation = [None] * (hidden_length + 1)
        for i in range(hidden_length):
            self.activation[i] = get_nonlinearity(self.activation_name)
        self.activation[-1] = get_nonlinearity(self.output_activation_name)

    def _get_linear_layer(self):
        node_list = [self.input_size] + self.hidden_layer_list + [self.output_size]
        self.linear_layer = [None] * (len(node_list) - 1)
        for i in range(len(self.linear_layer)):
            fc = nn.Linear(node_list[i], node_list[i+1])
            setattr(self, f"linear_layer_{i}", fc)
            self.linear_layer[i] = fc

    def forward(self, inputs):
        x = inputs
        for linear, activation in self.layer:
            x = activation(linear(x))
        return x