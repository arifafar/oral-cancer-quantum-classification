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

from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    SamplerV2 as Sampler
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
    "ibm_fez_hardware_results.txt"
)

OUTPUT_PREDICTIONS = os.path.join(
    BASE_DIR,
    "ibm_fez_hardware_predictions.csv"
)

OUTPUT_COUNTS = os.path.join(
    BASE_DIR,
    "ibm_fez_hardware_counts.npz"
)

OUTPUT_JOB = os.path.join(
    BASE_DIR,
    "ibm_fez_hardware_job.txt"
)

BACKEND_NAME = "ibm_fez"

NUM_QUBITS = 4
NUM_LAYERS = 2

BATCH_SIZE = 16

# ------------------------------------------------------------
# IMPORTANT:
# 512 shots is used initially to conserve QPU time.
# ------------------------------------------------------------

SHOTS = 512


# ============================================================
# START TIMER
# ============================================================

total_start = time.time()


# ============================================================
# CHECK FILES
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CHECKING REQUIRED FILES"
)

print(
    "============================================================"
)

for filename in [
    PROJECTION_WEIGHTS,
    TRAINED_PARAMETERS
]:

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{filename}"
        )

    print(
        "Found:",
        os.path.basename(filename)
    )


# ============================================================
# LOAD PROJECTION WEIGHTS
# ============================================================

print(
    "\nLoading learned projection..."
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


# ============================================================
# LOAD TRAINED QUANTUM PARAMETERS
# ============================================================

print(
    "\nLoading trained quantum parameters..."
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
        "Quantum weights must have shape (2,4,3)."
    )


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print(
    "\nLoading validation dataset..."
)

val_datagen = (
    tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
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
# RESNET50V2
# ============================================================

print(
    "\nLoading ResNet50V2..."
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

features = resnet(
    inputs,
    training=False
)

gap = GlobalAveragePooling2D()(
    features
)

feature_model = Model(
    inputs=inputs,
    outputs=gap
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


# ============================================================
# EXTRACT VALIDATION FEATURES
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "EXTRACTING VALIDATION FEATURES"
)

print(
    "============================================================"
)

val_generator.reset()

all_features = []
all_labels = []

for batch_index in range(
    len(val_generator)
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
        f"\rProcessed "
        f"{batch_index + 1}/"
        f"{len(val_generator)}",
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


# ============================================================
# APPLY LEARNED PROJECTION
# ============================================================

features_4d = projection_model.predict(
    features_2048,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(
    "\n4-D feature shape:",
    features_4d.shape
)


# ============================================================
# CONNECT TO IBM QUANTUM
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CONNECTING TO IBM QUANTUM"
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

status = backend.status()

print(
    "Backend:",
    backend.name
)

print(
    "Qubits:",
    backend.num_qubits
)

print(
    "Operational:",
    status.operational
)

print(
    "Pending jobs:",
    status.pending_jobs
)

if not status.operational:

    raise RuntimeError(
        "IBM FEZ is currently not operational."
    )


# ============================================================
# BUILD HARDWARE CIRCUIT
# ============================================================

def build_circuit(
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
        # Rot(phi, theta, omega)
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
        # CORRECT PENNYLANE DEFAULT RANGES
        #
        # Layer 1 -> range 1
        # Layer 2 -> range 2
        # ----------------------------------------------------

        entangling_range = (
            layer + 1
        ) % NUM_QUBITS

        for qubit in range(NUM_QUBITS):

            target = (
                qubit +
                entangling_range
            ) % NUM_QUBITS

            qc.cx(
                qubit,
                target
            )

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    for qubit in range(NUM_QUBITS):

        qc.measure(
            qubit,
            qubit
        )

    return qc


# ============================================================
# BUILD 150 CIRCUITS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "BUILDING HARDWARE CIRCUITS"
)

print(
    "============================================================"
)

circuits = []

for index in range(
    len(features_4d)
):

    circuit = build_circuit(
        features_4d[index],
        quantum_weights
    )

    circuits.append(
        circuit
    )

    print(
        f"\rBuilt "
        f"{index + 1}/"
        f"{len(features_4d)}",
        end=""
    )

print()


# ============================================================
# TRANSPILE FOR IBM FEZ
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "TRANSPILING FOR IBM FEZ"
)

print(
    "============================================================"
)

pass_manager = generate_preset_pass_manager(
    optimization_level=1,
    backend=backend
)

transpiled_circuits = []

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

print()


# ============================================================
# CHECK TRANSPILATION
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
            for instruction in circuit.data
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
    depths.min()
)

print(
    "Maximum:",
    depths.max()
)

print(
    "Mean:",
    depths.mean()
)

print(
    "\nTwo-qubit gates:"
)

print(
    "Minimum:",
    two_qubit_counts.min()
)

print(
    "Maximum:",
    two_qubit_counts.max()
)

print(
    "Mean:",
    two_qubit_counts.mean()
)


# ============================================================
# IMPORTANT FINAL CHECK BEFORE SUBMISSION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL HARDWARE SUBMISSION CHECK"
)

print(
    "============================================================"
)

print(
    f"Backend: {backend.name}"
)

print(
    f"Circuits: {len(transpiled_circuits)}"
)

print(
    f"Shots per circuit: {SHOTS}"
)

print(
    f"Maximum circuit depth: {depths.max()}"
)

print(
    f"Maximum two-qubit gates: "
    f"{two_qubit_counts.max()}"
)

print(
    "\nThe next command will submit ONE job "
    "to the real IBM FEZ QPU."
)

print(
    "This WILL consume QPU time."
)

print(
    "\n"
    "============================================================"
)


# ============================================================
# SUBMIT ONE SAMPLER JOB
# ============================================================

print(
    "\nSubmitting hardware job..."
)

hardware_start = time.time()

sampler = Sampler(
    mode=backend
)

job = sampler.run(
    transpiled_circuits,
    shots=SHOTS
)

job_id = job.job_id()

print(
    "\n"
    "============================================================"
)

print(
    "IBM HARDWARE JOB SUBMITTED"
)

print(
    "============================================================"
)

print(
    "Job ID:",
    job_id
)

print(
    "Backend:",
    backend.name
)

print(
    "Shots per circuit:",
    SHOTS
)

print(
    "Number of circuits:",
    len(transpiled_circuits)
)

print(
    "\nWaiting for hardware result..."
)


# ============================================================
# SAVE JOB ID IMMEDIATELY
# ============================================================

with open(
    OUTPUT_JOB,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"IBM FEZ hardware job ID: {job_id}\n"
    )

    f.write(
        f"Backend: {backend.name}\n"
    )

    f.write(
        f"Circuits: {len(transpiled_circuits)}\n"
    )

    f.write(
        f"Shots per circuit: {SHOTS}\n"
    )


# ============================================================
# WAIT FOR RESULT
# ============================================================

result = job.result()

hardware_time = (
    time.time() -
    hardware_start
)

print(
    "\nHardware job completed."
)

print(
    "Wall-clock execution/wait time:",
    f"{hardware_time:.2f}",
    "seconds"
)


# ============================================================
# EXTRACT COUNTS
# ============================================================

print(
    "\nExtracting hardware counts..."
)

all_counts = []

for index in range(
    len(transpiled_circuits)
):

    pub_result = result[index]

    counts = (
        pub_result
        .data
        .meas
        .get_counts()
    )

    all_counts.append(
        counts
    )

    print(
        f"\rProcessed "
        f"{index + 1}/"
        f"{len(transpiled_circuits)}",
        end=""
    )

print()


# ============================================================
# SAVE COUNTS
# ============================================================

np.savez(
    OUTPUT_COUNTS,
    counts=np.array(
        all_counts,
        dtype=object
    )
)

print(
    "\nHardware counts saved:"
)

print(
    OUTPUT_COUNTS
)


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

        bitstring = (
            bitstring.replace(
                " ",
                ""
            )
        )

        # Qiskit bit ordering:
        # rightmost bit = classical bit 0
        bit = int(
            bitstring[
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

    return (
        expectation /
        shots
    )


# ============================================================
# CALCULATE FOUR Z EXPECTATIONS
# ============================================================

quantum_outputs = []

for index, counts in enumerate(
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

quantum_outputs = np.array(
    quantum_outputs,
    dtype=np.float64
)

print(
    "\nQuantum output shape:",
    quantum_outputs.shape
)


# ============================================================
# APPLY FIXED CLASSICAL CLASSIFIER
# ============================================================

raw_logits = (
    np.dot(
        quantum_outputs,
        classifier_weights
    )
    +
    classifier_bias
)

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
# DISPLAY FINAL HARDWARE RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "REAL IBM FEZ HARDWARE RESULTS"
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

print(
    "\nIBM Job ID:",
    job_id
)


# ============================================================
# GET QPU USAGE
# ============================================================

print(
    "\nTrying to retrieve IBM QPU usage..."
)

quantum_seconds = None

try:

    metrics = job.metrics()

    print(
        "\nJob metrics:"
    )

    print(
        metrics
    )

    usage = metrics.get(
        "usage",
        {}
    )

    quantum_seconds = usage.get(
        "quantum_seconds"
    )

    if quantum_seconds is not None:

        print(
            "\nQPU quantum time:",
            quantum_seconds,
            "seconds"
        )

except Exception as error:

    print(
        "\nCould not retrieve QPU usage automatically."
    )

    print(
        "Reason:",
        error
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_data = np.column_stack(
    [
        np.arange(len(labels)),
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
# SAVE FINAL RESULTS
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
        "REAL IBM FEZ HARDWARE EXECUTION\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Backend: {backend.name}\n"
    )

    f.write(
        f"Job ID: {job_id}\n"
    )

    f.write(
        f"Number of circuits: "
        f"{len(transpiled_circuits)}\n"
    )

    f.write(
        f"Shots per circuit: {SHOTS}\n"
    )

    f.write(
        "Qubits: 4\n"
    )

    f.write(
        "Variational layers: 2\n"
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
        "Entangling ranges: [1, 2]\n\n"
    )

    f.write(
        "Transpilation:\n"
    )

    f.write(
        f"Depth minimum: {depths.min()}\n"
    )

    f.write(
        f"Depth maximum: {depths.max()}\n"
    )

    f.write(
        f"Depth mean: {depths.mean():.2f}\n"
    )

    f.write(
        f"Two-qubit gates minimum: "
        f"{two_qubit_counts.min()}\n"
    )

    f.write(
        f"Two-qubit gates maximum: "
        f"{two_qubit_counts.max()}\n"
    )

    f.write(
        f"Two-qubit gates mean: "
        f"{two_qubit_counts.mean():.2f}\n\n"
    )

    f.write(
        "Hardware results:\n"
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
        "\n\n"
    )

    f.write(
        f"QPU quantum seconds: "
        f"{quantum_seconds}\n"
    )

    f.write(
        f"Hardware wall-clock time: "
        f"{hardware_time:.4f} seconds\n"
    )

    f.write(
        f"Total script time: "
        f"{total_time:.4f} seconds\n"
    )


# ============================================================
# FINISHED
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "REAL IBM FEZ EXPERIMENT COMPLETE"
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
    "\nJob ID:",
    job_id
)

print(
    "\nResults:"
)

print(
    OUTPUT_RESULTS
)

print(
    "\nPredictions:"
)

print(
    OUTPUT_PREDICTIONS
)

print(
    "\n============================================================"
)