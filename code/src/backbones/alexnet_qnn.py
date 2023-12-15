from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector


def parity(x):
    return "{:b}".format(x).count("1") % 2

def create_qnn(output_dim, backend):
    feature_map = ZZFeatureMap(output_dim)
    ansatz = RealAmplitudes(output_dim, reps=1)
    qc = QuantumCircuit(output_dim)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # qnn = SamplerQNN(
    #     circuit=qc,
    #     input_params=feature_map.parameters,
    #     weight_params=ansatz.parameters,
    #     interpret=parity,
    #     output_shape=output_dim,
    # )

    qnn = CircuitQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=parity,
        output_shape=output_dim,
        quantum_instance=backend,
    )

    return qnn

class AlexNetQNN(nn.Module):
    def __init__(self, output_dim, backend=None, is_qnn=True):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3, 64, 3, 2, 1
            ),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 16 * 16, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

        self.is_qnn = is_qnn
        self.hybrid = TorchConnector(create_qnn(output_dim, backend))

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        if self.is_qnn:
            x = self.hybrid(x)
        return x, h
