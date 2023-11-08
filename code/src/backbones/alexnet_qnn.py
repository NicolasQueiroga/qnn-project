from torch import nn
import torch

from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


def create_qnn(output_dim):
    feature_map = ZZFeatureMap(output_dim)
    ansatz = RealAmplitudes(output_dim, reps=1)
    qc = QuantumCircuit(output_dim)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn

class AlexNetQNN(nn.Module):
    def __init__(self, output_dim, is_qnn=True):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2),  # kernel_size
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
            nn.ReLU(inplace=True)
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
        self.qantum_features = nn.Sequential(
            TorchConnector(create_qnn(output_dim)),
            nn.Linear(1, 4),
        )

    def forward(self, x, ibmq_backend=None):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        if self.is_qnn:
            if ibmq_backend is not None:
                self.qantum_features[0].backend = ibmq_backend
            x = self.qantum_features(x)
        return x, h