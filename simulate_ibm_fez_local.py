import os
import time
import h5py
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet_v2 import (
    ResNet50V2,
    preprocess_input
)
from tensorflow.keras.layers import (
    Dense,
    Input,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score
)

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

# ============================================================
# QISKIT AER
# ============================================================

try:
    from qiskit_aer import AerSimulator
except ImportError:
    raise ImportError(
        "\nqiskit-aer is not installed.\n\n"
        "Install it with:\n"
        "pip install qiskit-aer\n"
    )


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\oral cancer\Oral Cancer Dataset_Split"

VAL_DIR = os.path.join(
    BASE_DIR,
    "val"
)

PROJECTION_WEIGHTS = os.path.join(
    BASE_DIR,
    "hybrid_learned_projection.weights.h5"
)

TRAINED_PARAMETERS = os.path.join(
    BASE_DIR,
    "hybrid_trained_parameters.npz"
)

OUTPUT_RESULTS = os.path.join(
    BASE_DIR,
    "ibm_fez_local_noisy_results.txt"
)

OUTPUT_PREDICTIONS = os.path.join(
    BASE_DIR,
    "ibm_fez_local_noisy_predictions.csv"
)

OUTPUT_COUNTS = os.path.join(
    BASE_DIR,
    "ibm_fez_local_noisy_counts.npz"
)

NUM_QUBITS = 4
NUM_LAYERS = 2

BATCH_SIZE = 16

# ------------------------------------------------------------
# Shots used for local noisy simulation
# ------------------------------------------------------------

SHOTS = 1024

BACKEND_NAME = "ibm_fez"


# ============================================================
# START TIMER
# ============================================================

total_start = time.time()


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

required_files = [
    PROJECTION_WEIGHTS,
    TRAINED_PARAMETERS
]

for filename in required_files:

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{filename}"
        )

    print(
        "Found:",
        os.path.basename(filename)
    )


# ============================================================
# LOAD LEARNED PROJECTION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "LOADING LEARNED 2048 -> 4 PROJECTION"
)

print(
    "============================================================"
)

with h5py.File(
    PROJECTION_WEIGHTS,
    "r"
) as f:

    projection_kernel = np.array(
        f["layers/dense/vars/0"],
        dtype=np.float32
    )

    projection_bias = np.array(
        f["layers/dense/vars/1"],
        dtype=np.float32
    )

print(
    "Projection kernel:",
    projection_kernel.shape
)

print(
    "Projection bias:",
    projection_bias.shape
)

if projection_kernel.shape != (2048, 4):

    raise RuntimeError(
        "Unexpected projection kernel shape."
    )

if projection_bias.shape != (4,):

    raise RuntimeError(
        "Unexpected projection bias shape."
    )


# ============================================================
# LOAD TRAINED QUANTUM PARAMETERS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "LOADING TRAINED HYBRID PARAMETERS"
)

print(
    "============================================================"
)

params = np.load(
    TRAINED_PARAMETERS
)

quantum_weights = params[
    "quantum_weights"
]

classifier_weights = params[
    "classifier_weights"
]

classifier_bias = params[
    "classifier_bias"
]

print(
    "Quantum weights:",
    quantum_weights.shape
)

print(
    "Classifier weights:",
    classifier_weights.shape
)

print(
    "Classifier bias:",
    classifier_bias.shape
)

if quantum_weights.shape != (2, 4, 3):

    raise RuntimeError(
        f"Expected quantum weights (2,4,3), "
        f"got {quantum_weights.shape}"
    )

if classifier_weights.shape != (4, 2):

    raise RuntimeError(
        f"Expected classifier weights (4,2), "
        f"got {classifier_weights.shape}"
    )

if classifier_bias.shape != (2,):

    raise RuntimeError(
        f"Expected classifier bias (2,), "
        f"got {classifier_bias.shape}"
    )


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "LOADING VALIDATION DATA"
)

print(
    "============================================================"
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print(
    "Validation images:",
    val_generator.samples
)

print(
    "Class indices:",
    val_generator.class_indices
)


# ============================================================
# RESNET50V2 FEATURE EXTRACTOR
# ============================================================

print(
    "\n"
    "Loading ResNet50V2..."
)

resnet = ResNet50V2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

resnet.trainable = False

inputs = Input(
    shape=(224, 224, 3)
)

resnet_features = resnet(
    inputs,
    training=False
)

gap = GlobalAveragePooling2D()(
    resnet_features
)

feature_model = Model(
    inputs=inputs,
    outputs=gap
)

print(
    "ResNet50V2 feature extractor ready."
)


# ============================================================
# LEARNED 2048 -> 4 PROJECTION
# ============================================================

projection_input = Input(
    shape=(2048,)
)

projection_layer = Dense(
    4,
    activation="tanh"
)

projection_output = projection_layer(
    projection_input
)

projection_model = Model(
    projection_input,
    projection_output
)

projection_layer.set_weights(
    [
        projection_kernel,
        projection_bias
    ]
)

print(
    "Learned projection weights loaded."
)


# ============================================================
# EXTRACT VALIDATION FEATURES
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "EXTRACTING RESNET50V2 VALIDATION FEATURES"
)

print(
    "============================================================"
)

val_generator.reset()

all_features = []
all_labels = []

num_batches = len(
    val_generator
)

for batch_index in range(
    num_batches
):

    images, labels_batch = (
        val_generator[batch_index]
    )

    batch_features = (
        feature_model.predict(
            images,
            verbose=0
        )
    )

    all_features.append(
        batch_features
    )

    all_labels.append(
        labels_batch
    )

    print(
        f"\rProcessed batch "
        f"{batch_index + 1}/{num_batches}",
        end=""
    )

print()

features_2048 = np.concatenate(
    all_features,
    axis=0
)

labels = np.concatenate(
    all_labels,
    axis=0
).astype(int)

print(
    "2048-D feature shape:",
    features_2048.shape
)

print(
    "Labels shape:",
    labels.shape
)


# ============================================================
# APPLY LEARNED PROJECTION
# ============================================================

print(
    "\nApplying learned 2048 -> 4 projection..."
)

features_4d = projection_model.predict(
    features_2048,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(
    "4-D feature shape:",
    features_4d.shape
)

print(
    "4-D feature minimum:",
    np.min(features_4d)
)

print(
    "4-D feature maximum:",
    np.max(features_4d)
)


# ============================================================
# CONNECT TO IBM QUANTUM
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "LOADING IBM FEZ NOISE MODEL"
)

print(
    "============================================================"
)

service = QiskitRuntimeService(
    channel="ibm_quantum_platform"
)

backend = service.backend(
    BACKEND_NAME
)

print(
    "Backend:",
    backend.name
)

print(
    "Backend qubits:",
    backend.num_qubits
)

print(
    "Backend operational:",
    backend.status().operational
)

print(
    "Pending jobs:",
    backend.status().pending_jobs
)

print(
    "\nIMPORTANT:"
)

print(
    "No QPU job will be submitted."
)

print(
    "The IBM FEZ noise model will be simulated locally."
)


# ============================================================
# CREATE LOCAL IBM FEZ NOISY SIMULATOR
# ============================================================

print(
    "\nCreating local IBM FEZ noisy simulator..."
)

noisy_simulator = AerSimulator.from_backend(
    backend
)

print(
    "Local noisy simulator ready."
)


# ============================================================
# BUILD QISKIT CIRCUIT
# ============================================================

def build_qiskit_circuit(
    input_vector,
    weights
):

    qc = QuantumCircuit(
        NUM_QUBITS,
        NUM_QUBITS
    )

    # --------------------------------------------------------
    # AngleEmbedding RX
    # --------------------------------------------------------

    for qubit in range(NUM_QUBITS):

        qc.rx(
            float(input_vector[qubit]),
            qubit
        )

    # --------------------------------------------------------
    # StronglyEntanglingLayers
    # --------------------------------------------------------

    for layer in range(NUM_LAYERS):

        # ----------------------------------------------------
        # PennyLane Rot(phi, theta, omega)
        #
        # RZ(phi) RY(theta) RZ(omega)
        # ----------------------------------------------------

        for qubit in range(NUM_QUBITS):

            phi = float(
                weights[
                    layer,
                    qubit,
                    0
                ]
            )

            theta = float(
                weights[
                    layer,
                    qubit,
                    1
                ]
            )

            omega = float(
                weights[
                    layer,
                    qubit,
                    2
                ]
            )

            qc.rz(
                phi,
                qubit
            )

            qc.ry(
                theta,
                qubit
            )

            qc.rz(
                omega,
                qubit
            )

        # ----------------------------------------------------
        # CORRECT DEFAULT STRONGLY ENTANGLING RANGES
        #
        # Layer 1 -> range 1
        # Layer 2 -> range 2
        #
        # Layer 1:
        # 0 -> 1
        # 1 -> 2
        # 2 -> 3
        # 3 -> 0
        #
        # Layer 2:
        # 0 -> 2
        # 1 -> 3
        # 2 -> 0
        # 3 -> 1
        # ----------------------------------------------------

        entangling_range = (
            layer + 1
        ) % NUM_QUBITS

        for qubit in range(NUM_QUBITS):

            target = (
                qubit + entangling_range
            ) % NUM_QUBITS

            qc.cx(
                qubit,
                target
            )

    # --------------------------------------------------------
    # Measure each qubit into corresponding classical bit
    # --------------------------------------------------------

    for qubit in range(NUM_QUBITS):

        qc.measure(
            qubit,
            qubit
        )

    return qc


# ============================================================
# CREATE ALL 150 CIRCUITS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "BUILDING 4-QUBIT CIRCUITS"
)

print(
    "============================================================"
)

circuits = []

for sample_index in range(
    len(features_4d)
):

    qc = build_qiskit_circuit(
        features_4d[sample_index],
        quantum_weights
    )

    circuits.append(
        qc
    )

    print(
        f"\rBuilt circuit "
        f"{sample_index + 1}/"
        f"{len(features_4d)}",
        end=""
    )

print()


# ============================================================
# TRANSPILATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "TRANSPILING CIRCUITS"
)

print(
    "============================================================"
)

pass_manager = generate_preset_pass_manager(
    optimization_level=1,
    backend=backend
)

transpiled_circuits = []

transpile_start = time.time()

for index, circuit in enumerate(
    circuits
):

    transpiled = pass_manager.run(
        circuit
    )

    transpiled_circuits.append(
        transpiled
    )

    print(
        f"\rTranspiled "
        f"{index + 1}/"
        f"{len(circuits)}",
        end=""
    )

transpile_time = (
    time.time() -
    transpile_start
)

print()

print(
    "\nTranspilation time:",
    f"{transpile_time:.2f}",
    "seconds"
)


# ============================================================
# CHECK CIRCUIT DEPTH
# ============================================================

depths = np.array(
    [
        circuit.depth()
        for circuit in transpiled_circuits
    ]
)

two_qubit_counts = np.array(
    [
        sum(
            1
            for instruction
            in circuit.data
            if len(
                instruction.qubits
            ) == 2
        )
        for circuit in transpiled_circuits
    ]
)

print(
    "\nTranspiled depth:"
)

print(
    "Minimum:",
    np.min(depths)
)

print(
    "Maximum:",
    np.max(depths)
)

print(
    "Mean:",
    np.mean(depths)
)

print(
    "\nTwo-qubit gates:"
)

print(
    "Minimum:",
    np.min(two_qubit_counts)
)

print(
    "Maximum:",
    np.max(two_qubit_counts)
)

print(
    "Mean:",
    np.mean(two_qubit_counts)
)


# ============================================================
# LOCAL NOISY EXECUTION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "RUNNING IBM FEZ NOISE MODEL LOCALLY"
)

print(
    "============================================================"
)

print(
    "Number of circuits:",
    len(transpiled_circuits)
)

print(
    "Shots per circuit:",
    SHOTS
)

print(
    "\nThis is LOCAL simulation."
)

print(
    "NO IBM QPU TIME IS BEING USED."
)

simulation_start = time.time()

job = noisy_simulator.run(
    transpiled_circuits,
    shots=SHOTS
)

print(
    "\nSimulation job submitted locally."
)

result = job.result()

simulation_time = (
    time.time() -
    simulation_start
)

print(
    "\nLocal noisy simulation time:",
    f"{simulation_time:.2f}",
    "seconds"
)


# ============================================================
# EXTRACT COUNTS
# ============================================================

print(
    "\nExtracting measurement counts..."
)

all_counts = []

for sample_index in range(
    len(transpiled_circuits)
):

    counts = result.get_counts(
        sample_index
    )

    all_counts.append(
        counts
    )

    print(
        f"\rProcessed counts "
        f"{sample_index + 1}/"
        f"{len(transpiled_circuits)}",
        end=""
    )

print()


# ============================================================
# COMPUTE Z EXPECTATIONS
# ============================================================

def expectation_from_counts(
    counts,
    qubit,
    shots
):

    expectation = 0.0

    for bitstring, count in counts.items():

        # Remove possible whitespace
        bitstring_clean = (
            bitstring.replace(
                " ",
                ""
            )
        )

        # Qiskit classical-bit ordering:
        # rightmost bit = qubit 0
        bit = int(
            bitstring_clean[
                -(qubit + 1)
            ]
        )

        if bit == 0:

            eigenvalue = 1.0

        else:

            eigenvalue = -1.0

        expectation += (
            eigenvalue *
            count
        )

    return expectation / shots


# ============================================================
# CALCULATE FOUR Z EXPECTATIONS
# ============================================================

quantum_outputs = []

for sample_index, counts in enumerate(
    all_counts
):

    z_values = []

    for qubit in range(
        NUM_QUBITS
    ):

        z = expectation_from_counts(
            counts,
            qubit,
            SHOTS
        )

        z_values.append(z)

    quantum_outputs.append(
        z_values
    )

    print(
        f"\rCalculated expectations "
        f"{sample_index + 1}/"
        f"{len(all_counts)}",
        end=""
    )

print()

quantum_outputs = np.array(
    quantum_outputs,
    dtype=np.float64
)

print(
    "\nQuantum output shape:",
    quantum_outputs.shape
)

print(
    "Quantum output range:",
    np.min(quantum_outputs),
    "to",
    np.max(quantum_outputs)
)


# ============================================================
# APPLY FIXED CLASSICAL CLASSIFIER
# ============================================================

print(
    "\nApplying trained classical output layer..."
)

raw_logits = (
    np.dot(
        quantum_outputs,
        classifier_weights
    )
    +
    classifier_bias
)

# Stable softmax
exp_logits = np.exp(
    raw_logits -
    np.max(
        raw_logits,
        axis=1,
        keepdims=True
    )
)

probabilities = (
    exp_logits /
    np.sum(
        exp_logits,
        axis=1,
        keepdims=True
    )
)

predicted_labels = np.argmax(
    probabilities,
    axis=1
)

positive_probabilities = (
    probabilities[:, 1]
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    labels,
    predicted_labels
)

precision = precision_score(
    labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    labels,
    predicted_labels
)

cm = confusion_matrix(
    labels,
    predicted_labels
)

try:

    roc_auc = roc_auc_score(
        labels,
        positive_probabilities
    )

except ValueError:

    roc_auc = float("nan")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "IBM FEZ LOCAL NOISY SIMULATION RESULTS"
)

print(
    "============================================================"
)

print(
    f"Accuracy:           {accuracy:.4f}"
)

print(
    f"Weighted Precision: {precision:.4f}"
)

print(
    f"Weighted Recall:    {recall:.4f}"
)

print(
    f"Weighted F1:        {f1:.4f}"
)

print(
    f"Balanced Accuracy:  {balanced_acc:.4f}"
)

print(
    f"ROC-AUC:            {roc_auc:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_indices = np.arange(
    len(labels)
)

prediction_data = np.column_stack(
    [
        prediction_indices,
        labels,
        predicted_labels,
        probabilities[:, 0],
        probabilities[:, 1]
    ]
)

np.savetxt(
    OUTPUT_PREDICTIONS,
    prediction_data,
    delimiter=",",
    header=(
        "sample_index,true_label,"
        "predicted_label,prob_class_0,"
        "prob_class_1"
    ),
    comments="",
    fmt=[
        "%d",
        "%d",
        "%d",
        "%.8f",
        "%.8f"
    ]
)

print(
    "\nPredictions saved:"
)

print(
    OUTPUT_PREDICTIONS
)


# ============================================================
# SAVE COUNTS
# ============================================================

# Counts are saved as individual arrays using object dtype.

np.savez(
    OUTPUT_COUNTS,
    counts=np.array(
        all_counts,
        dtype=object
    )
)

print(
    "Measurement counts saved:"
)

print(
    OUTPUT_COUNTS
)


# ============================================================
# SAVE RESULTS
# ============================================================

total_time = (
    time.time() -
    total_start
)

with open(
    OUTPUT_RESULTS,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "IBM FEZ LOCAL NOISY SIMULATION\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        "IMPORTANT: This is a LOCAL noisy "
        "simulation using the IBM FEZ noise model.\n"
    )

    f.write(
        "No IBM hardware job was submitted.\n"
    )

    f.write(
        "No QPU time was consumed.\n\n"
    )

    f.write(
        f"Backend noise model: {BACKEND_NAME}\n"
    )

    f.write(
        f"Qubits: {NUM_QUBITS}\n"
    )

    f.write(
        f"Variational layers: {NUM_LAYERS}\n"
    )

    f.write(
        "Embedding: AngleEmbedding RX\n"
    )

    f.write(
        "Ansatz: StronglyEntanglingLayers\n"
    )

    f.write(
        "Entangler: CNOT\n"
    )

    f.write(
        "Entangling ranges: [1, 2]\n"
    )

    f.write(
        f"Shots: {SHOTS}\n\n"
    )

    f.write(
        "Transpilation statistics:\n"
    )

    f.write(
        f"Depth minimum: {np.min(depths)}\n"
    )

    f.write(
        f"Depth maximum: {np.max(depths)}\n"
    )

    f.write(
        f"Depth mean: {np.mean(depths):.2f}\n"
    )

    f.write(
        f"Two-qubit gates minimum: "
        f"{np.min(two_qubit_counts)}\n"
    )

    f.write(
        f"Two-qubit gates maximum: "
        f"{np.max(two_qubit_counts)}\n"
    )

    f.write(
        f"Two-qubit gates mean: "
        f"{np.mean(two_qubit_counts):.2f}\n\n"
    )

    f.write(
        "Results:\n"
    )

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Weighted Precision: {precision:.6f}\n"
    )

    f.write(
        f"Weighted Recall: {recall:.6f}\n"
    )

    f.write(
        f"Weighted F1: {f1:.6f}\n"
    )

    f.write(
        f"Balanced Accuracy: {balanced_acc:.6f}\n"
    )

    f.write(
        f"ROC-AUC: {roc_auc:.6f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nTiming:\n"
    )

    f.write(
        f"Transpilation time: "
        f"{transpile_time:.4f} seconds\n"
    )

    f.write(
        f"Simulation time: "
        f"{simulation_time:.4f} seconds\n"
    )

    f.write(
        f"Total time: "
        f"{total_time:.4f} seconds\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "LOCAL IBM FEZ NOISY SIMULATION COMPLETE"
)

print(
    "============================================================"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Weighted F1: {f1:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_acc:.4f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_RESULTS
)

print(
    "\nNO IBM HARDWARE JOB WAS SUBMITTED."
)

print(
    "NO QPU TIME WAS CONSUMED."
)

print(
    "============================================================"
)