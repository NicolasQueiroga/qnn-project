from src.backbones import alexnet_qnn
import torch.nn as nn

def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)

def get_model(model_name, output_dim, backend=None, is_qnn=True, init_params=True):
    if model_name == 'alexnet_qnn':
        model = alexnet_qnn.AlexNetQNN(output_dim, backend=backend, is_qnn=is_qnn)
    else:
        raise NotImplementedError

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')

    if init_params:
        model.apply(initialize_parameters)

    return model

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True