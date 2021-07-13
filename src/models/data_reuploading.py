import cirq
import math
import sympy

import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq


def one_qubit_unitary(bit, inputs):
    """Make a Cirq circuit enacting a rotation of the bloch sphere about the X,
    Y and Z axis, that depends on the values in `inputs`.

    Args:
        bit (cirq.Qubit): Qubit to be acted on.
        inputs (List[double]): Rotation angles for unitary gate.

    Returns:
        [cirq.Circuit]: 1-qubit unitary gate.
    """
    return cirq.Circuit(
        cirq.X(bit) ** inputs[0],
        cirq.Y(bit) ** inputs[1],
        cirq.Z(bit) ** inputs[2],
    )


class ReuploadingCircuit(tf.keras.layers.Layer):
    """
    Construct a multi-qubit data re-uploading classifier as specified in:
        https://arxiv.org/pdf/1907.02085.pdf.

    Args:
        input_dims (integer): Number of input features. If this is not a
                              multiple of 3, it will be zero-padded.
        num_qubits (integer): Number of qubits in the circuit.
        num_layers (integer): Number of re-uploading layers.
    """

    def __init__(self, input_dims, num_qubits, num_layers, name='re-uploading_circuit'):
        super(ReuploadingCircuit, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Encode inputs in groups of 3
        self.num_encoding_blocks = math.ceil(input_dims / 3)
        self.num_params = self.num_encoding_blocks * 3

        weights_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.num_weights = self.num_qubits * self.num_layers * self.num_params
        self.thetas = tf.Variable(
            initial_value=weights_init(
                shape=(1, self.num_weights), dtype='float32'),
            trainable=True,
            name='thetas')
        self.w = tf.Variable(
            initial_value=weights_init(
                shape=(1, self.num_weights), dtype='float32'),
            trainable=True,
            name='w')
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

        self.pqc = self.build_model()

    def build_model(self):
        qubits = cirq.GridQubit.rect(1, self.num_qubits)
        readouts = [cirq.Z(qubits[-1])]

        # Sympy symbols for encoding inputs
        encoding_params = sympy.symbols(f'x_0:{self.num_weights}')
        encoding_params = np.asarray(encoding_params).reshape(
            (self.num_layers, self.num_qubits, self.num_encoding_blocks, 3))

        # Define explicit symbol order to follow the alphabetical order of their symbol names,
        # as processed by the ControlledPQC.
        symbols = [str(symb) for symb in list(encoding_params.flat)]
        self.indices = tf.constant([sorted(symbols).index(a) for a in symbols])

        # Create circuit
        circuit = cirq.Circuit()
        for l in range(self.num_layers):
            # Encoding layer
            for i, q in enumerate(qubits):
                circuit += cirq.Circuit(one_qubit_unitary(
                    q, encoding_params[l, i, b]) for b in range(self.num_encoding_blocks))

            # Entangling layer
            if self.num_qubits > 1:
                if l < self.num_layers - 1:
                    # entangle qubits if not last layer
                    circuit += [cirq.CZ(q0, q1)
                                for q0, q1 in zip(qubits, qubits[1:])]

        pqc = tfq.layers.ControlledPQC(circuit, readouts)

        return pqc

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Duplicate thetas across batch
        thetas = tf.tile(self.thetas, multiples=[
                         batch_size, 1], name='tile_weights')
        w = tf.tile(self.w, multiples=[
            batch_size, 1], name='tile_input_weights')

        # Pad tensors if necessary
        num_padding = self.num_params - inputs.shape[1]
        if num_padding != 0:
            inputs = tf.pad(inputs, tf.constant(
                [[0, 0], [0, num_padding]]), name='pad_inputs')

        # Duplicate inputs across qubits and layers
        inputs = tf.tile(inputs, tf.constant(
            [1, self.num_qubits * self.num_layers]), name='pad_inputs')

        # Scale inputs
        inputs = tf.math.multiply(inputs, w, name='scale_inputs')
        inputs = tf.math.add(inputs, thetas, name='shift_inputs')
        inputs = tf.keras.activations.tanh(inputs)

        # Flatten inputs
        flattened = tf.keras.layers.Flatten(name='flatten')(inputs)

        circuits = tf.repeat(self.empty_circuit,
                             repeats=batch_size, name='tile_circuits')
        model_params = tf.concat(
            [flattened, thetas], axis=-1, name='concat_model_params')
        model_params = tf.gather(
            model_params, self.indices, axis=-1, name='permute_model_params')

        return self.pqc([circuits, model_params])


def create_reuploading_model(num_features, num_qubits, depth, include_classical=False):
    inputs = tf.keras.Input(shape=(num_features,), dtype=tf.dtypes.float32)
    x = ReuploadingCircuit(num_features, num_qubits, depth)(inputs)

    if include_classical:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        # Rescale from [-1, 1] to [0, 1]
        out = (x + 1.0) / 2.0

    return tf.keras.Model(inputs=inputs, outputs=out)
