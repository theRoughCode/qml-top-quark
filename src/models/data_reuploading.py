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
        cirq.rx(inputs[0])(bit),
        cirq.ry(inputs[1])(bit),
        cirq.rz(inputs[2])(bit)
    )


class ReuploadingCircuit(tf.keras.layers.Layer):
    def __init__(self, num_features, num_qubits, depth, batch_size):
        super(ReuploadingCircuit, self).__init__()
        self.num_encoding_blocks = math.ceil(num_features / 3)
        self.num_params = self.num_encoding_blocks * 3
        self.num_qubits = num_qubits
        self.num_weights = self.num_qubits * depth * 3
        self.depth = depth
        self.batch_size = batch_size
        self.q_weights = tf.Variable(
            initial_value=np.random.uniform(
                0, 2 * np.pi, (1, self.num_weights)),
            dtype="float32",
            trainable=True)
        self.pqc = self.build_model()

    def build_model(self):
        qubits = cirq.GridQubit.rect(1, self.num_qubits)
        out_bits = qubits[-1:]
        # 1-local operators to read out
        readouts = [cirq.Z(bit) for bit in out_bits]

        circuit = cirq.Circuit()
        encoding_params = sympy.symbols(
            'enc0:{}'.format(self.num_encoding_blocks * 3))
        trainable_params = sympy.symbols('proc0:{}'.format(self.num_weights))

        for depth in range(self.depth):
            for qIdx in range(self.num_qubits):
                # data re-uploading
                for i in range(self.num_encoding_blocks):
                    circuit += one_qubit_unitary(qubits[qIdx],
                                                 encoding_params[i * 3:(i + 1) * 3])
                # trainable processing
                circuit += one_qubit_unitary(qubits[qIdx],
                                             trainable_params[:3])
                trainable_params = trainable_params[3:]

            # entangle qubits if not last layer
            if self.num_qubits > 1 and depth < self.depth - 1:
                for qIdx in range(depth % 2, self.num_qubits, 2):
                    circuit += cirq.CZ(qubits[qIdx],
                                       qubits[(qIdx+1) % self.num_qubits])

        pqc = tfq.layers.ControlledPQC(circuit, readouts)

        return pqc

    def call(self, inputs):
        # Duplicate weights across batch
        weights = tf.tile(
            self.q_weights, (self.batch_size, 1), name='tile_weights')
        # Pad tensors if necessary
        num_padding = self.num_params - inputs.shape[1]
        if num_padding != 0:
            inputs = tf.pad(inputs, tf.constant([[0, 0, ], [0, num_padding]]))
        # Flatten sub-image
        flattened = tf.keras.layers.Flatten(name='flatten')(inputs)
        # Prepare inputs
        input_q_circ = tfq.convert_to_tensor(
            [cirq.Circuit() for _ in range(self.batch_size)])
        model_params = tf.concat(
            [flattened, weights], axis=-1, name='concat_model_params')
        out = self.pqc([input_q_circ, model_params])
        return out


def create_model(num_features, num_qubits, depth, batch_size, include_classical=False):
    inputs = tf.keras.Input(shape=(num_features,), dtype=tf.dtypes.float32)
    x = ReuploadingCircuit(num_features, num_qubits, depth, batch_size)(inputs)

    if include_classical:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        out = (x + 1.0) / 2.0

    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model
