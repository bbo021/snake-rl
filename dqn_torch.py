import torch
import torch.nn as nn
import json

"""
Reference-code to building the model from json in original tensorflow agent.py:

input_board is some tmp Input() from tensorflow to pass through the net
x = input_board
for layer in m['model']:
    l = m['model'][layer]
    if('Conv2D' in layer):
        # add convolutional layer
        x = Conv2D(**l)(x)
    if('Flatten' in layer):
        x = Flatten()(x)
    if('Dense' in layer):
        x = Dense(**l)(x)
out = Dense(self._n_actions, activation='linear', name='action_values')(x)
model = Model(inputs=input_board, outputs=out)
model.compile(optimizer=RMSprop(0.0005), loss=mean_huber_loss)
"""

class DQN(nn.Module):
    """
    Deep-Q NN to represent the model in /model_config/v17.1.json
    
    
    """
    def __init__(self, version, board_size, n_frames, n_actions):
        super().__init__()

        self.version = version
        self.board_size = board_size
        self.n_frames = n_frames
        self.n_actions = n_actions

        # define the input layer, shape is dependent on the board size and frames
        with open('model_config/{:s}.json'.format(self.version), 'r') as f:
            m = json.loads(f.read())
        
        conv, linear = [], []
        channels = n_frames
        for layer in m['model']: # Name of layer
            layer_i = m['model'][layer] # Contents/structure of current layer in model

            if 'Conv2D' in layer: # Extract Conv2D layers, keep in mind data-format channels_last in TF, permute x later
                filters = layer_i['filters']
                kernel_size = layer_i['kernel_size']
                kernel_w, kernel_h = kernel_size[0], kernel_size[1]

                padding = layer_i.get('padding', None)
                activation = layer_i.get('activation', None)
                pad_w = pad_h = 0
                activation_function = None
                if padding:
                    pad_w = self._port_tf_padding_to_torch(padding, kernel_w)
                    pad_h = self._port_tf_padding_to_torch(padding, kernel_h)

                if activation == 'relu':
                    activation_function = nn.ReLU()

                conv2d = nn.Conv2d(
                    channels,
                    filters,
                    kernel_size=kernel_size,
                    padding=(pad_h, pad_w)
                )
                channels = filters
                
                conv.append(conv2d)
                if activation_function is not None:
                    conv.append(activation_function)
            elif 'Dense' in layer: # Extract linear layer, though the features are not known yet, so we keep info
                linear.append(layer_i)
            elif 'Flatten' in layer: # Extract flattening of conv (which is empty in model 17.1)
                conv.append(nn.Flatten()) 
        
        self.conv_block = nn.Sequential(
            *conv
        )

        with torch.no_grad():
            input_board = torch.zeros(1, n_frames, board_size, board_size)
            out_board = self.conv_block(input_board)
            features = out_board.view(1, -1).shape[1]
            linear_torch = []
            for lin in linear:
                units = lin['units']
                
                linear_torch.append(nn.Linear(features, units))
                if lin.get('activation', None) == 'relu':
                    linear_torch.append(nn.ReLU())

                features = units

        self.linear_block = nn.Sequential(
            *linear_torch
        )

        # Final output layer mapping features to the actions of the agent
        self.out = nn.Linear(features, n_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # Since TF format is channels_last, we permute so N H W C becomes N C H W
        x = self.conv_block(x)
        x = self.linear_block(x)

        return self.out(x)

    def _port_tf_padding_to_torch(self, tf_padding, kernel_size):
        """
        Converts Tensorflow padding to torch
        
        :param tf_padding: Tensorflow padding argument
        :param kernel_size: Scalar (not tuple)
        """
        if tf_padding == 'same':
            return (kernel_size - 1) // 2
        else:
            return 0