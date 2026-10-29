import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = r"D:\oral cancer\Oral Cancer Dataset_Split"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
SEED = 42


# Output files
MODEL_PATH = os.path.join(
    BASE_DIR,
    "oral_cancer_resnet50v2_classical.keras"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "classical_resnet50v2_results.txt"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "classical_resnet50v2_history.json"
)

CM_PATH = os.path.join(
    BASE_DIR,
    "classical_resnet50v2_confusion_matrix.png"
)

ROC_PATH = os.path.join(
    BASE_DIR,
    "classical_resnet50v2_roc_curve.png"
)

LOSS_PATH = os.path.join(
    BASE_DIR,
    "classical_resnet50v2_loss.png"
)

ACC_PATH = os.path.join(
    BASE_DIR,
    "classical_resnet50v2_accuracy.png"
)


# ============================================================
# 2. CHECK FOLDERS
# ============================================================

print("\n" + "=" * 70)
print("CLASSICAL RESNET50V2 ORAL CANCER CLASSIFICATION")
print("=" * 70)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Validation folder not found:\n{VAL_DIR}"
    )

print("\nDataset:")
print("Training folder   :", TRAIN_DIR)
print("Validation folder :", VAL_DIR)


# ============================================================
# 3. DATA PREPARATION
# ============================================================

print("\nPreparing image generators...")


# Training augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True
)


# Validation - no augmentation
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)


val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


print("\nClass indices:")
print(train_generator.class_indices)

print("\nTraining images   :", train_generator.samples)
print("Validation images:", val_generator.samples)


# ============================================================
# 4. BUILD RESNET50V2
# ============================================================

print("\nLoading pretrained ResNet50V2...")


base_model = ResNet50V2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Freeze pretrained convolutional backbone
base_model.trainable = False


# Feature extraction
x = base_model.output

x = GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


# Classical classification layers
x = Dense(
    256,
    activation="relu",
    name="dense_256"
)(x)

x = Dropout(
    0.5,
    name="dropout"
)(x)


# Binary classification
output = Dense(
    2,
    activation="softmax",
    name="classification"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# 5. COMPILE
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel created successfully.")

print("\nTotal parameters:",
      model.count_params())


# ============================================================
# 6. MODEL CHECKPOINT
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 7. TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

start_training = time.time()


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    verbose=1
)


training_time = time.time() - start_training


print("\nTraining completed.")

print(
    f"Training time: {training_time:.2f} seconds"
)

print(
    f"Training time: {training_time / 60:.2f} minutes"
)


# ============================================================
# 8. SAVE TRAINING HISTORY
# ============================================================

history_dict = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(HISTORY_PATH, "w") as file:
    json.dump(
        history_dict,
        file,
        indent=4
    )


# ============================================================
# 9. LOAD BEST MODEL
# ============================================================

print("\nLoading best ResNet50V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# 10. VALIDATION PREDICTION
# ============================================================

print("\nEvaluating on validation set...")

val_generator.reset()

start_inference = time.time()


probabilities = model.predict(
    val_generator,
    verbose=1
)


inference_time = time.time() - start_inference


predicted_labels = np.argmax(
    probabilities,
    axis=1
)

true_labels = val_generator.classes


inference_per_image = (
    inference_time /
    val_generator.samples
)


# ============================================================
# 11. CLASS INFORMATION
# ============================================================

class_indices = val_generator.class_indices

print("\nClass indices:")
print(class_indices)


# Find Cancer class
cancer_class = None

for class_name, class_index in class_indices.items():

    if class_name.lower() == "cancer":
        cancer_class = class_index


if cancer_class is None:

    raise ValueError(
        "Cancer class was not found. "
        "Check your folder names."
    )


# Other class
non_cancer_class = 1 - cancer_class


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


# Reorder matrix:
#
#              Predicted
#              Cancer  Non-Cancer
#
# Cancer          TP       FN
# Non-Cancer      FP       TN


cm_ordered = np.array([
    [
        cm[cancer_class, cancer_class],
        cm[cancer_class, non_cancer_class]
    ],
    [
        cm[non_cancer_class, cancer_class],
        cm[non_cancer_class, non_cancer_class]
    ]
])


TP = cm_ordered[0, 0]
FN = cm_ordered[0, 1]
FP = cm_ordered[1, 0]
TN = cm_ordered[1, 1]


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n              Predicted")
print("              Cancer  Non-Cancer")
print(
    f"Cancer       {TP:6d}  {FN:10d}"
)
print(
    f"Non-Cancer   {FP:6d}  {TN:10d}"
)


# ============================================================
# 13. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


precision = precision_score(
    true_labels,
    predicted_labels,
    pos_label=cancer_class,
    zero_division=0
)


sensitivity = recall_score(
    true_labels,
    predicted_labels,
    pos_label=cancer_class,
    zero_division=0
)


specificity = (
    TN / (TN + FP)
    if (TN + FP) > 0
    else 0
)


f1 = f1_score(
    true_labels,
    predicted_labels,
    pos_label=cancer_class,
    zero_division=0
)


balanced_accuracy = (
    sensitivity + specificity
) / 2


# ============================================================
# 14. ROC-AUC
# ============================================================

cancer_probability = probabilities[:, cancer_class]


binary_true = (
    true_labels == cancer_class
).astype(int)


roc_auc = roc_auc_score(
    binary_true,
    cancer_probability
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

target_names = [
    class_name
    for class_name, index
    in sorted(
        class_indices.items(),
        key=lambda x: x[1]
    )
]


report = classification_report(
    true_labels,
    predicted_labels,
    target_names=target_names,
    digits=4,
    zero_division=0
)


# ============================================================
# 16. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLASSICAL RESNET50V2 RESULTS")
print("=" * 70)

print(f"\nTP                  : {TP}")
print(f"FN                  : {FN}")
print(f"FP                  : {FP}")
print(f"TN                  : {TN}")

print(
    f"\nAccuracy            : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision           : "
    f"{precision * 100:.2f}%"
)

print(
    f"Sensitivity/Recall  : "
    f"{sensitivity * 100:.2f}%"
)

print(
    f"Specificity         : "
    f"{specificity * 100:.2f}%"
)

print(
    f"F1-score            : "
    f"{f1 * 100:.2f}%"
)

print(
    f"Balanced Accuracy   : "
    f"{balanced_accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC             : "
    f"{roc_auc:.4f}"
)

print(
    f"\nTraining time       : "
    f"{training_time:.2f} seconds"
)

print(
    f"Inference time      : "
    f"{inference_time:.2f} seconds"
)

print(
    f"Inference/image     : "
    f"{inference_per_image * 1000:.2f} ms"
)


print("\nClassification Report:")
print(report)

print("=" * 70)


# ============================================================
# 17. SAVE TEXT RESULTS
# ============================================================

with open(RESULTS_PATH, "w") as file:

    file.write(
        "CLASSICAL RESNET50V2 ORAL CANCER "
        "CLASSIFICATION RESULTS\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        "Evaluation set: Validation set\n\n"
    )

    file.write(
        f"Training images    : "
        f"{train_generator.samples}\n"
    )

    file.write(
        f"Validation images  : "
        f"{val_generator.samples}\n\n"
    )

    file.write("Confusion Matrix\n")
    file.write("----------------\n")
    file.write(
        "              Cancer  Non-Cancer\n"
    )
    file.write(
        f"Cancer        {TP:6d}  {FN:10d}\n"
    )
    file.write(
        f"Non-Cancer    {FP:6d}  {TN:10d}\n\n"
    )

    file.write(f"TP: {TP}\n")
    file.write(f"FN: {FN}\n")
    file.write(f"FP: {FP}\n")
    file.write(f"TN: {TN}\n\n")

    file.write(
        f"Accuracy           : "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision          : "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Sensitivity/Recall : "
        f"{sensitivity * 100:.2f}%\n"
    )

    file.write(
        f"Specificity        : "
        f"{specificity * 100:.2f}%\n"
    )

    file.write(
        f"F1-score           : "
        f"{f1 * 100:.2f}%\n"
    )

    file.write(
        f"Balanced Accuracy  : "
        f"{balanced_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"ROC-AUC            : "
        f"{roc_auc:.4f}\n\n"
    )

    file.write(
        f"Training time      : "
        f"{training_time:.2f} seconds\n"
    )

    file.write(
        f"Inference/image    : "
        f"{inference_per_image * 1000:.2f} ms\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "---------------------\n"
    )

    file.write(report)


# ============================================================
# 18. CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    cm_ordered,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - ResNet50V2"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Cancer", "Non-Cancer"]
)

plt.yticks(
    [0, 1],
    ["Cancer", "Non-Cancer"]
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm_ordered[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )


plt.tight_layout()

plt.savefig(
    CM_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    binary_true,
    cancer_probability
)


plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"ResNet50V2 (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Classical ResNet50V2"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    ROC_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. LOSS CURVE
# ============================================================

plt.figure(figsize=(7, 6))

plt.plot(
    history.history["loss"],
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss - ResNet50V2"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    LOSS_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. ACCURACY CURVE
# ============================================================

plt.figure(figsize=(7, 6))

plt.plot(
    history.history["accuracy"],
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy - ResNet50V2"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ACC_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. FINISH
# ============================================================

print("\n" + "=" * 70)
print("CLASSICAL RESNET50V2 EXPERIMENT COMPLETED")
print("=" * 70)

print("\nSaved files:")

print("\nModel:")
print(MODEL_PATH)

print("\nResults:")
print(RESULTS_PATH)

print("\nConfusion matrix:")
print(CM_PATH)

print("\nROC curve:")
print(ROC_PATH)

print("\nLoss curve:")
print(LOSS_PATH)

print("\nAccuracy curve:")
print(ACC_PATH)

print("\nExperiment completed successfully.")