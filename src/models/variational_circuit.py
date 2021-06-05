from quple.circuits.variational_circuits import EfficientSU2
from quple.classifiers import VQC
from quple.data_encoding import FirstOrderPauliZEncoding

import cirq
import tensorflow as tf


def create_variational_circuit(num_qubits, include_classical=False, depth=2):
    # First order pauli z encoding circuit with depth 2
    encoding_circuit = FirstOrderPauliZEncoding(
        feature_dimension=num_qubits, copies=depth)

    # Efficient SU2 variational circuit with depth 2
    variational_circuit = EfficientSU2(n_qubit=num_qubits, copies=2)
    variational_circuit.add_readout('XX')

    # Construct the VQC model
    vqc = VQC(encoding_circuit, variational_circuit,
              activation='sigmoid',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['binary_accuracy', 'AUC'],
              readout=[cirq.Z(variational_circuit.readout_qubit)],
              classical_layer=include_classical,
              loss='mse')

    return vqc
