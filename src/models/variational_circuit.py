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


def FirstOrderPauliZEncoding(qubits, encoding_params, copies=1):
    circuit = cirq.Circuit()
    for _ in range(copies):
        for qubit, input in zip(qubits, encoding_params):
            circuit += cirq.H(qubit)
            circuit += cirq.Z(qubit) ** input
    return circuit


def SecondOrderPauliZEncoding(qubits, rotation_params, entangling_params, copies=1):
    circuit = cirq.Circuit()
    circuit += [cirq.H(q) for q in qubits]

    for _ in range(copies):
        for qubit, input in zip(qubits, rotation_params):
            circuit += cirq.Z(qubit) ** input

        for q1, q2, param in zip(qubits, qubits[1:], entangling_params):
            circuit += cirq.CNOT(q1, q2)
            circuit += cirq.Z(q2) ** param
            circuit += cirq.CNOT(q1, q2)

    return circuit


def EfficientSU2(qubits, ansatz_params, copies=1):
    circuit = cirq.Circuit()
    for l in range(copies + 1):
        circuit += cirq.Circuit(cirq.Y(q) **
                                ansatz_params[l, i, 0] for i, q in enumerate(qubits))
        circuit += cirq.Circuit(cirq.Z(q) **
                                ansatz_params[l, i, 1] for i, q in enumerate(qubits))

        # Entanglement strategy
        if l < copies:
            circuit += [cirq.CNOT(q0, q1)
                        for q0, q1 in zip(qubits, qubits[1:])]

    return circuit


class VQC(tf.keras.layers.Layer):
    """Create a variational quantum circuit.

    Args:
        num_qubits (integer): Number of qubits.
        encoding_copies (integer): Number of encoding repetitions.
        ansatz_copies (integer): Number of variational repetitions.
        name (str, optional): Model name. Defaults to 'vqc'.
    """

    def __init__(self, num_qubits, encoding_copies, ansatz_copies, name='vqc'):
        super(VQC, self).__init__()
        self.num_qubits = num_qubits
        self.encoding_copies = encoding_copies
        self.ansatz_copies = ansatz_copies

        self.num_ansatz_params = self.num_qubits * (ansatz_copies + 1) * 2

        weights_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.num_weights = self.num_ansatz_params
        self.thetas = tf.Variable(
            initial_value=weights_init(
                shape=(1, self.num_weights), dtype='float32'),
            trainable=True,
            name='thetas')
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

        self.pqc = self.build_model()

    def build_model(self):
        qubits = cirq.GridQubit.rect(1, self.num_qubits)
        readouts = [cirq.Z(readout) for readout in qubits]

        # Sympy symbols for encoding inputs
        encoding_params = sympy.symbols(f'x_0:{self.num_qubits}')
        encoding_params = np.asarray(encoding_params)

        # Sympy symbols for trainable parameters
        ansatz_params = sympy.symbols('θ_0:{}'.format(self.num_ansatz_params))
        ansatz_params = np.asarray(ansatz_params).reshape(
            (self.ansatz_copies + 1, self.num_qubits, 2))

        # Define explicit symbol order to follow the alphabetical order of their symbol names,
        # as processed by the ControlledPQC.
        symbols = [str(symb) for symb in list(
            encoding_params.flat) + list(ansatz_params.flat)]
        self.indices = tf.constant([sorted(symbols).index(a) for a in symbols])

        # Create circuit
        circuit = cirq.Circuit()
        # Encoding layer
        circuit += FirstOrderPauliZEncoding(qubits,
                                            encoding_params, copies=self.encoding_copies)
        # Variational layer
        circuit += EfficientSU2(qubits, ansatz_params,
                                copies=self.ansatz_copies)

        pqc = tfq.layers.ControlledPQC(circuit, readouts)

        return pqc

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Duplicate thetas across batch
        thetas = tf.tile(self.thetas, multiples=[
                         batch_size, 1], name='tile_weights')

        circuits = tf.repeat(self.empty_circuit,
                             repeats=batch_size, name='tile_circuits')
        model_params = tf.concat(
            [inputs, thetas], axis=-1, name='concat_model_params')
        model_params = tf.gather(
            model_params, self.indices, axis=-1, name='permute_model_params')

        return self.pqc([circuits, model_params])


def create_vqc_model(num_features, encoding_copies, ansatz_copies, with_classical=False):
    inputs = tf.keras.Input(shape=(num_features,), dtype=tf.dtypes.float32)
    x = VQC(num_features, encoding_copies, ansatz_copies)(inputs)

    if with_classical:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        out = (x + 1.) / 2.

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model
