# Oral Cancer Quantum–Classical Classification

This repository contains Python scripts for the oral cancer image-classification experiments described in the associated research work. The scripts cover a classical ResNet50V2 baseline, a hybrid quantum–classical model, local noisy simulation using an IBM FEZ backend noise model, and execution on IBM FEZ hardware.

## Repository contents

- `classical_resent50v2(5).py` — trains/evaluates the classical ResNet50V2 baseline.
- `hybrid_train_corrected(4).py` — trains the ResNet50V2 feature extractor/projection and four-qubit hybrid quantum classifier.
- `simulate_ibm_fez_local(7).py` — performs a local noisy simulation using an IBM FEZ backend-derived noise model; the script states that it does not submit a QPU job.
- `run_ibm_fez_hardware(6).py` — submits circuits to real IBM FEZ hardware.
- `run_ibm_fez_hardware_timed(1).py` — hardware execution script with additional timing and job metadata collection.
- `requirements.txt` — Python package dependencies.

## Method overview

The scripts use ResNet50V2 pretrained on ImageNet to extract image features. The hybrid pipeline reduces the 2048-dimensional pooled feature vector to four dimensions using a learned dense projection with a `tanh` activation. A four-qubit circuit encodes these features with RX rotations, applies trainable variational layers, and measures Pauli-Z expectation values. A classical output layer produces the class probabilities.

The baseline script is configured for two classes. The hybrid and hardware scripts are also configured for two classes; check that the folder names and class ordering match the experiment before running.

## Dataset and directory setup

The scripts currently use a local Windows dataset path:

`D:\oral cancer\Oral Cancer Dataset_Split`

The expected directory structure is:

```text
Oral Cancer Dataset_Split/
├── train/
│   ├── class_1/
│   └── class_2/
└── val/
    ├── class_1/
    └── class_2/
```

Replace the `BASE_DIR` value in each script with the path to your own dataset split. The class subfolder names must match the labels used in your experiment. Do not upload private or restricted patient images to this repository.

## Installation

Use a dedicated Python environment and install a compatible Python version for TensorFlow, PennyLane, and Qiskit on your operating system.

```bash
python -m venv .venv
```

On Windows:

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Package compatibility can vary by Python version and operating system. The dependency list is a starting point, not a verified lockfile. For reproducibility, record the exact package versions from the environment used to generate the reported results.

## Required trained artifacts

The noisy-simulation and hardware scripts expect trained model artifacts to exist in the dataset directory, including:

- `hybrid_learned_projection.weights.h5`
- `hybrid_trained_parameters.npz`

These files are generated or saved by the training workflow. Ensure they are present and correspond to the same trained experiment before running simulation or hardware scripts. Do not upload trained artifacts unless you have checked their size, licensing, and whether you intend to make them public.

## Running the scripts

Run the classical baseline:

```bash
python "classical_resent50v2(5).py"
```

Train the hybrid model:

```bash
python "hybrid_train_corrected(4).py"
```

Run the local noisy simulation:

```bash
python "simulate_ibm_fez_local(7).py"
```

Run a real IBM FEZ hardware experiment:

```bash
python "run_ibm_fez_hardware(6).py"
```

Or run the timed hardware version:

```bash
python "run_ibm_fez_hardware_timed(1).py"
```

The hardware scripts require an IBM Quantum account with access to the `ibm_fez` backend and a working local Qiskit Runtime account configuration. They submit jobs to real quantum hardware and consume quantum-processing resources. Run them only when you intend to submit a job and have checked the expected circuit count and shots.

## Reproducibility notes

- The scripts define random seeds, but exact results may still vary with software versions, hardware calibration, backend availability, and shot noise.
- The code uses local absolute paths that must be updated for another machine.
- The dependency list is not yet pinned to exact versions. Before claiming exact reproducibility, record the Python and package versions used in the original experiments.
- Confirm the validation set and class ordering before comparing metrics across scripts.

## Data and code use

This repository shares experiment code only. The dataset is not included. Users should obtain the dataset from its authorized original source and follow its terms of use.

## Citation

If you use this code in academic work, cite the associated paper once its bibliographic details are available. Please do not interpret results on the retrospective dataset as evidence of clinical readiness without independent external and prospective validation.
