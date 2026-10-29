import os
import time
import numpy as np
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = r"D:\oral cancer\Oral Cancer Dataset_Split"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

NUM_QUBITS = 4
NUM_LAYERS = 2
NUM_CLASSES = 2

PROJECTION_EPOCHS = 20
QUANTUM_EPOCHS = 30

LEARNING_RATE_PROJECTION = 0.001
LEARNING_RATE_QUANTUM = 0.01

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
pnp.random.seed(SEED)


# ============================================================
# 2. CHECK DATASET
# ============================================================

class_labels = sorted([
    folder for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])

print("\nDetected classes:", class_labels)

if len(class_labels) != 2:
    raise ValueError(
        f"Expected exactly 2 classes, but found {len(class_labels)}: {class_labels}"
    )


# ============================================================
# 3. DATA GENERATORS
# ============================================================

# IMPORTANT:
# ResNet50V2 ImageNet weights are used with preprocess_input,
# rather than simple rescaling by 1/255.

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

print("\nClass indices:")
print(train_generator.class_indices)

print("\nTraining images:", train_generator.samples)
print("Validation images:", val_generator.samples)


# ============================================================
# 4. RESNET50V2 FEATURE EXTRACTION
# ============================================================

print("\n==============================================")
print("Extracting ResNet50V2 Features")
print("==============================================")

start_feature_time = time.time()

base_model = ResNet50V2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the pretrained backbone.
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

feature_extractor = Model(
    inputs=inputs,
    outputs=x
)

X_train_deep = feature_extractor.predict(
    train_generator,
    verbose=1
)

X_val_deep = feature_extractor.predict(
    val_generator,
    verbose=1
)

y_train = train_generator.classes.astype(np.int64)
y_val = val_generator.classes.astype(np.int64)

print("\nResNet50V2 feature shape:")
print("Training:", X_train_deep.shape)
print("Validation:", X_val_deep.shape)

print("\nExpected feature dimension after GAP: 2048")

feature_extraction_time = time.time() - start_feature_time

print(
    f"\nFeature extraction time: "
    f"{feature_extraction_time:.2f} seconds"
)


# ============================================================
# 5. CLASS WEIGHTS
# ============================================================

unique_classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(unique_classes, class_weights_array)
}

print("\nClass weights:")
print(class_weights)


# ============================================================
# 6. LEARNED FOUR-DIMENSIONAL PROJECTION
# ============================================================

print("\n==============================================")
print("Training 4-Dimensional Feature Projection")
print("==============================================")

# The original code created Dense(4) but never trained it.
# Here, the Dense(4) projection is actually learned.

projection_input = Input(shape=(X_train_deep.shape[1],))

projection_output = Dense(
    NUM_QUBITS,
    activation="tanh",
    name="learned_quantum_projection"
)(projection_input)

projection_classifier = Dense(
    NUM_CLASSES,
    activation="softmax",
    name="projection_classifier"
)(projection_output)

projection_model = Model(
    inputs=projection_input,
    outputs=projection_classifier
)

projection_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_PROJECTION
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

projection_start = time.time()

projection_history = projection_model.fit(
    X_train_deep,
    y_train,
    validation_data=(X_val_deep, y_val),
    epochs=PROJECTION_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    verbose=1
)

projection_time = time.time() - projection_start

print(
    f"\nProjection training time: "
    f"{projection_time:.2f} seconds"
)


# ============================================================
# 7. CREATE LEARNED 4-D FEATURE EXTRACTOR
# ============================================================

learned_projection = Model(
    inputs=projection_model.input,
    outputs=projection_model.get_layer(
        "learned_quantum_projection"
    ).output
)

X_train_4d = learned_projection.predict(
    X_train_deep,
    batch_size=BATCH_SIZE,
    verbose=1
)

X_val_4d = learned_projection.predict(
    X_val_deep,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nLearned 4-D feature shapes:")
print("Training:", X_train_4d.shape)
print("Validation:", X_val_4d.shape)

print("\nFirst five training feature vectors:")
print(X_train_4d[:5])


# ============================================================
# 8. QUANTUM CIRCUIT
# ============================================================

print("\n==============================================")
print("Building 4-Qubit Quantum Circuit")
print("==============================================")

dev = qml.device(
    "default.qubit",
    wires=NUM_QUBITS
)


@qml.qnode(
    dev,
    interface="autograd"
)
def quantum_circuit(inputs, weights):

    # Four classical features -> four qubits
    qml.AngleEmbedding(
        inputs,
        wires=range(NUM_QUBITS),
        rotation="X"
    )

    # Two strongly entangling variational layers
    qml.StronglyEntanglingLayers(
        weights,
        wires=range(NUM_QUBITS)
    )

    # Pauli-Z expectation values
    return [
        qml.expval(qml.PauliZ(i))
        for i in range(NUM_QUBITS)
    ]


# ============================================================
# 9. INITIALIZE QUANTUM PARAMETERS
# ============================================================

pnp.random.seed(SEED)

quantum_weights = pnp.array(
    0.01 * pnp.random.randn(
        NUM_LAYERS,
        NUM_QUBITS,
        3
    ),
    requires_grad=True
)

# Classical output layer:
# 4 quantum features -> 2 output classes
classifier_weights = pnp.array(
    0.01 * pnp.random.randn(
        NUM_QUBITS,
        NUM_CLASSES
    ),
    requires_grad=True
)

classifier_bias = pnp.array(
    np.zeros(NUM_CLASSES),
    requires_grad=True
)

print("\nQuantum weight shape:")
print(quantum_weights.shape)

print("\nTotal quantum trainable parameters:")
print(
    NUM_LAYERS *
    NUM_QUBITS *
    3
)

print("\nExpected:")
print("2 × 4 × 3 = 24")


# ============================================================
# 10. QUANTUM PREDICTION
# ============================================================

def quantum_features(
    features,
    weights
):

    outputs = []

    for feature_vector in features:

        result = quantum_circuit(
            feature_vector,
            weights
        )

        outputs.append(result)

    return pnp.stack(outputs)


def hybrid_predict(
    features,
    q_weights,
    clf_weights,
    clf_bias
):

    q_output = quantum_features(
        features,
        q_weights
    )

    logits = pnp.dot(
        q_output,
        clf_weights
    ) + clf_bias

    # Stable softmax
    logits_shifted = (
        logits -
        pnp.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_logits = pnp.exp(
        logits_shifted
    )

    probabilities = (
        exp_logits /
        pnp.sum(
            exp_logits,
            axis=1,
            keepdims=True
        )
    )

    return probabilities


# ============================================================
# 11. WEIGHTED LOSS
# ============================================================

def weighted_cross_entropy(
    q_weights,
    clf_weights,
    clf_bias,
    features,
    targets,
    sample_weights
):

    predictions = hybrid_predict(
        features,
        q_weights,
        clf_weights,
        clf_bias
    )

    one_hot = pnp.eye(
        NUM_CLASSES
    )[targets]

    predictions = pnp.clip(
        predictions,
        1e-8,
        1.0
    )

    losses = -pnp.sum(
        one_hot *
        pnp.log(predictions),
        axis=1
    )

    weighted_losses = (
        losses *
        sample_weights
    )

    # Normalize by total sample weight
    return (
        pnp.sum(weighted_losses) /
        pnp.sum(sample_weights)
    )


# ============================================================
# 12. TRAINING SAMPLE WEIGHTS
# ============================================================

train_sample_weights = np.array([
    class_weights[int(label)]
    for label in y_train
])

train_sample_weights = pnp.array(
    train_sample_weights,
    requires_grad=False
)

val_sample_weights = pnp.ones(
    len(y_val)
)


# ============================================================
# 13. QUANTUM TRAINING
# ============================================================

print("\n==============================================")
print("Training Hybrid Quantum Model")
print("==============================================")

optimizer = qml.AdamOptimizer(
    stepsize=LEARNING_RATE_QUANTUM
)

best_val_loss = float("inf")

best_q_weights = quantum_weights.copy()
best_clf_weights = classifier_weights.copy()
best_clf_bias = classifier_bias.copy()

training_start = time.time()

train_loss_history = []
val_loss_history = []

train_accuracy_history = []
val_accuracy_history = []


for epoch in range(QUANTUM_EPOCHS):

    epoch_start = time.time()

    def cost_function(
        q_w,
        c_w,
        c_b
    ):

        return weighted_cross_entropy(
            q_w,
            c_w,
            c_b,
            X_train_4d,
            y_train,
            train_sample_weights
        )

    (
        quantum_weights,
        classifier_weights,
        classifier_bias
    ) = optimizer.step(
        cost_function,
        quantum_weights,
        classifier_weights,
        classifier_bias
    )

    # --------------------------------------------------------
    # Training loss
    # --------------------------------------------------------

    train_loss = weighted_cross_entropy(
        quantum_weights,
        classifier_weights,
        classifier_bias,
        X_train_4d,
        y_train,
        train_sample_weights
    )

    # --------------------------------------------------------
    # Validation loss
    # --------------------------------------------------------

    val_loss = weighted_cross_entropy(
        quantum_weights,
        classifier_weights,
        classifier_bias,
        X_val_4d,
        y_val,
        pnp.ones(len(y_val))
    )

    # --------------------------------------------------------
    # Training predictions
    # --------------------------------------------------------

    train_probabilities = hybrid_predict(
        X_train_4d,
        quantum_weights,
        classifier_weights,
        classifier_bias
    )

    train_predictions = np.argmax(
        np.asarray(train_probabilities),
        axis=1
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    val_probabilities = hybrid_predict(
        X_val_4d,
        quantum_weights,
        classifier_weights,
        classifier_bias
    )

    val_predictions = np.argmax(
        np.asarray(val_probabilities),
        axis=1
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    train_loss_value = float(
        np.asarray(train_loss)
    )

    val_loss_value = float(
        np.asarray(val_loss)
    )

    train_loss_history.append(
        train_loss_value
    )

    val_loss_history.append(
        val_loss_value
    )

    train_accuracy_history.append(
        train_accuracy
    )

    val_accuracy_history.append(
        val_accuracy
    )

    epoch_time = time.time() - epoch_start

    print(
        f"Epoch {epoch + 1:02d}/{QUANTUM_EPOCHS} | "
        f"Train Loss: {train_loss_value:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss_value:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Time: {epoch_time:.2f}s"
    )

    # --------------------------------------------------------
    # Save best model according to validation loss
    # --------------------------------------------------------

    if val_loss_value < best_val_loss:

        best_val_loss = val_loss_value

        best_q_weights = quantum_weights.copy()
        best_clf_weights = classifier_weights.copy()
        best_clf_bias = classifier_bias.copy()
        # ============================================================
# SAVE TRAINED PARAMETERS FOR IBM QUANTUM HARDWARE
# ============================================================

print("\nSaving trained hybrid model parameters...")

# Save learned 4-dimensional projection layer
learned_projection.save_weights(
    "hybrid_learned_projection.weights.h5"
)

# Save trained VQC parameters and classical classifier parameters
np.savez(
    "hybrid_trained_parameters.npz",
    quantum_weights=best_q_weights,
    classifier_weights=best_clf_weights,
    classifier_bias=best_clf_bias
)

print("Saved:")
print("  hybrid_learned_projection.weights.h5")
print("  hybrid_trained_parameters.npz")

print("Quantum parameter shape:", best_q_weights.shape)
print("Classifier weight shape:", best_clf_weights.shape)
print("Classifier bias shape:", best_clf_bias.shape)


total_training_time = (
    time.time() -
    training_start
)

print("\n==============================================")
print("Quantum Training Completed")
print("==============================================")

print(
    f"Quantum training time: "
    f"{total_training_time:.2f} seconds"
)

print(
    f"Quantum training time: "
    f"{total_training_time / 60:.2f} minutes"
)


# ============================================================
# 14. FINAL VALIDATION PREDICTION
# ============================================================

print("\n==============================================")
print("Final Hybrid Model Evaluation")
print("==============================================")

inference_start = time.time()

final_probabilities = hybrid_predict(
    X_val_4d,
    best_q_weights,
    best_clf_weights,
    best_clf_bias
)

final_probabilities = np.asarray(
    final_probabilities
)

inference_time = (
    time.time() -
    inference_start
)

y_pred = np.argmax(
    final_probabilities,
    axis=1
)

y_true = np.asarray(
    y_val,
    dtype=np.int64
)


# ============================================================
# 15. INFERENCE TIME
# ============================================================

num_validation_images = len(y_true)

time_per_image = (
    inference_time /
    num_validation_images
)

print(
    f"\nValidation images: "
    f"{num_validation_images}"
)

print(
    f"Total inference time: "
    f"{inference_time:.4f} seconds"
)

print(
    f"Inference time per image: "
    f"{time_per_image:.6f} seconds"
)

print(
    f"Inference latency per image: "
    f"{time_per_image * 1000:.2f} ms"
)


# ============================================================
# 16. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
)

print("\n==============================================")
print("FINAL HYBRID MODEL METRICS")
print("==============================================")

print(
    f"Accuracy       : {accuracy:.4f}"
)

print(
    f"Precision      : {precision:.4f}"
)

print(
    f"Recall         : {recall:.4f}"
)

print(
    f"F1-score       : {f1:.4f}"
)


# ============================================================
# 17. DETAILED CLASSIFICATION REPORT
# ============================================================

print("\n==============================================")
print("DETAILED CLASSIFICATION REPORT")
print("==============================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================")

print(cm)

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - 4-Qubit Hybrid Model"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_labels)
)

plt.xticks(
    tick_marks,
    class_labels,
    rotation=45
)

plt.yticks(
    tick_marks,
    class_labels
)

threshold = cm.max() / 2.0

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )

plt.ylabel(
    "True Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.tight_layout()

plt.savefig(
    "hybrid_4qubit_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 19. ROC-AUC
# ============================================================

try:

    # Probability of the second class
    y_probability = final_probabilities[:, 1]

    auc_score = roc_auc_score(
        y_true,
        y_probability
    )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_probability
    )

    print("\n==============================================")
    print("ROC-AUC")
    print("==============================================")

    print(
        f"ROC-AUC: {auc_score:.4f}"
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"Hybrid VQC (AUC = {auc_score:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - 4-Qubit Hybrid Model"
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        "hybrid_4qubit_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

except Exception as e:

    print(
        "\nROC-AUC could not be calculated:"
    )

    print(e)


# ============================================================
# 20. TRAINING CURVES
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    range(1, QUANTUM_EPOCHS + 1),
    train_loss_history,
    label="Training Loss"
)

plt.plot(
    range(1, QUANTUM_EPOCHS + 1),
    val_loss_history,
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Hybrid Model Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "hybrid_training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    range(1, QUANTUM_EPOCHS + 1),
    train_accuracy_history,
    label="Training Accuracy"
)

plt.plot(
    range(1, QUANTUM_EPOCHS + 1),
    val_accuracy_history,
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Hybrid Model Training and Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "hybrid_training_validation_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 21. FINAL PAPER SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RESULTS FOR MANUSCRIPT")
print("=" * 60)

print(
    f"Model                         : "
    f"ResNet50V2 + 4-Qubit VQC"
)

print(
    f"Number of qubits              : "
    f"{NUM_QUBITS}"
)

print(
    f"Variational layers            : "
    f"{NUM_LAYERS}"
)

print(
    f"Quantum parameters            : "
    f"{NUM_LAYERS * NUM_QUBITS * 3}"
)

print(
    f"Accuracy                      : "
    f"{accuracy:.4f}"
)

print(
    f"Weighted Precision            : "
    f"{precision:.4f}"
)

print(
    f"Weighted Recall               : "
    f"{recall:.4f}"
)

print(
    f"Weighted F1-score             : "
    f"{f1:.4f}"
)

print(
    f"Training time                 : "
    f"{total_training_time:.2f} seconds"
)

print(
    f"Inference time per image      : "
    f"{time_per_image * 1000:.2f} ms"
)

print("=" * 60)
print("Experiment completed.")
print("=" * 60)