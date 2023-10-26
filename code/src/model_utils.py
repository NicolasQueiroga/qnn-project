from src.backbones import alexnet, alexnet_qnn

def get_model(model_name, output_dim):
    if model_name == 'alexnet':
        model = alexnet.AlexNet(output_dim)
    elif model_name == 'alexnet_qnn':
        model = alexnet_qnn.AlexNetQNN(output_dim)
    else:
        raise NotImplementedError

    return model