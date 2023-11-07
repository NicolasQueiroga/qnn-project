from qiskit_ibm_provider import IBMProvider, least_busy
import os


class IBMBackend:
    def __init__(self):
        try:
            self.provider = IBMProvider()
        except:
            IBMProvider.save_account(token=os.environ.get('IBMQ_TOKEN'))
            self.provider = IBMProvider()

    def get_provider(self):
        return self.provider

    def get_backend(self, backend_name='ibmq_qasm_simulator'):
        return self.provider.get_backend(backend_name)

