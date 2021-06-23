import cirq
import quple
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq


def create_quple_vqc_model(num_qubits, include_classical=False, depth=2):
    """Create VQC using quple library.

    Args:
        num_qubits (integer): Number of qubits
        include_classical (bool, optional): Whether to include classical activation layer. Defaults to False.
        depth (int, optional): Number of encoding repetitions. Defaults to 2.

    Returns:
        tf.keras.Model: VQC model.
    """
    # First order pauli z encoding circuit with depth 2
    encoding_circuit = quple.data_encoding.FirstOrderPauliZEncoding(
        feature_dimension=num_qubits, copies=depth)

    # Efficient SU2 variational circuit with depth 2
    variational_circuit = quple.circuits.variational_circuits.EfficientSU2(
        n_qubit=num_qubits, copies=2)
    variational_circuit.add_readout('XX')

    # Construct the VQC model
    vqc = quple.classifiers.VQC(encoding_circuit, variational_circuit,
                                activation='sigmoid',
                                optimizer=tf.keras.optimizers.Adam(
                                    learning_rate=0.01),
                                metrics=['binary_accuracy', 'AUC'],
                                readout=[
                                    cirq.Z(variational_circuit.readout_qubit)],
                                classical_layer=include_classical,
                                loss='mse')

    return vqc


def FirstOrderPauliZEncoding(qubits, inputs, copies=1):
    circuit = cirq.Circuit()
    for _ in range(copies):
        for qubit, input in zip(qubits, inputs):
            circuit += cirq.H(qubit)
            circuit += cirq.Z(qubit) ** input
    return circuit


def EfficientSU2(qubits, inputs, copies=1):
    circuit = cirq.Circuit()
    for i in range(copies + 1):
        for qubit in qubits:
            circuit += cirq.Y(qubit) ** inputs[0]
            circuit += cirq.Z(qubit) ** inputs[1]
            inputs = inputs[2:]

        # Entanglement strategy
        if i < copies:
            for i in range(len(qubits) - 1):
                circuit += cirq.CNOT(qubits[i], qubits[i+1])

    return circuit


def readout_prep(qubits, readout, inputs):
    circuit = cirq.Circuit()
    for qubit, input in zip(qubits, inputs):
        circuit += cirq.XX(qubit, readout) ** input
    return circuit


class VQC(tf.keras.layers.Layer):
    def __init__(self, num_qubits, encoding_copies, ansatz_copies, batch_size):
        super(VQC, self).__init__()
        self.num_qubits = num_qubits
        self.encoding_copies = encoding_copies
        self.ansatz_copies = ansatz_copies
        self.num_ansatz_params = self.num_qubits * (ansatz_copies + 1) * 2
        self.num_readout_params = self.num_qubits
        self.num_weights = self.num_ansatz_params + self.num_readout_params
        self.batch_size = batch_size
        self.q_weights = tf.Variable(
            initial_value=np.random.uniform(-1, 1, (1, self.num_weights)),
            dtype="float32",
            trainable=True)
        self.pqc = self.build_model()

    def build_model(self):
        qubits = cirq.GridQubit.rect(1, self.num_qubits)
        readout = cirq.GridQubit(-1, -1)
        # 1-local operators to read out
        readouts = [cirq.Z(readout)]

        encoding_params = sympy.symbols('x0:{}'.format(self.num_qubits))
        ansatz_params = sympy.symbols(
            'ansatz0:{}'.format(self.num_ansatz_params))
        readout_params = sympy.symbols(
            'readout0:{}'.format(self.num_readout_params))

        circuit = cirq.Circuit()
        circuit += FirstOrderPauliZEncoding(qubits,
                                            encoding_params, copies=self.encoding_copies)
        circuit += EfficientSU2(qubits, ansatz_params,
                                copies=self.ansatz_copies)
        circuit += readout_prep(qubits, readout, readout_params)

        pqc = tfq.layers.ControlledPQC(circuit, readouts)

        return pqc

    def call(self, inputs):
        # Duplicate weights across batch
        weights = tf.tile(
            self.q_weights, (self.batch_size, 1), name='tile_weights')
        # Flatten sub-image
        flattened = tf.keras.layers.Flatten(name='flatten')(inputs)
        # Prepare inputs
        input_q_circ = tfq.convert_to_tensor(
            [cirq.Circuit() for _ in range(self.batch_size)])
        model_params = tf.concat(
            [flattened, weights], axis=-1, name='concat_model_params')
        out = self.pqc([input_q_circ, model_params])
        return out


def create_vqc_model(num_features, encoding_copies, ansatz_copies, batch_size, with_classical=False):
    inputs = tf.keras.Input(shape=(num_features,), dtype=tf.dtypes.float32)
    x = VQC(num_features, encoding_copies, ansatz_copies, batch_size)(inputs)

    if with_classical:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        out = (x + 1.) / 2.

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model
