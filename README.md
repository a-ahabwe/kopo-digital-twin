Here is a comprehensive README file for your GitHub repository, based on the analysis of your "KOPO: The Digital Twin" notebook.

---

# KOPO: The Digital Twin

## Neuro-dynamic Stability Mapping & ADHD Classification

This project implements a **"Digital Twin"** approach to computational neuroscience, using EEG data to model and simulate the dynamic stability of neural signals. By training individual neural networks (Digital Twins) on specific EEG channels, the notebook analyzes the inherent stability of brain regions in Neurotypical vs. ADHD subjects.

Additionally, the project includes a traditional machine learning pipeline for classifying ADHD based on extracted EEG features.

---

## Table of Contents

* [Overview]()
* [Key Features]()
* [The "Digital Twin" Concept]()
* [Installation & Dependencies]()
* [Usage]()
* [Methodology]()
* [Results]()

---

## Overview

The core objective of this notebook is to move beyond static feature extraction and model the **dynamical systems** governing brain activity.

1. **Classification Pipeline:** Standard ML models (SVM, Random Forest, Logistic Regression) distinguish ADHD from Control subjects using statistical features.
2. **Digital Twin Pipeline:** A lightweight Neural Network (`ChannelTwin`) is trained to predict the next time-step of a signal for every specific EEG channel. This "Twin" mimics the physics of that brain region. We then perturb the Twin to measure its **stability** (resistance to drift), generating a topographic "Stability Map" of the brain.

---

## Key Features

* **Signal Processing:** Butterworth bandpass filtering (0.5 - 50 Hz) and epoch generation with overlap.
* **Feature Engineering:** Extraction of statistical moments (Std Dev, Mean Power, Peak-to-Peak, etc.) per channel.
* **ML Classification:** Comparative analysis using:
* Logistic Regression
* Support Vector Machines (RBF Kernel)
* Random Forest


* **Digital Twin Modeling:** PyTorch-based neural networks trained to model channel-specific dynamics.
* **Brain Mapping:** Visualization of brain stability scores using 2D topographic head plots (neuro-imaging style interpolation).

---

## The "Digital Twin" Concept

Instead of just classifying data, we create a simulation of the data source.

1. **Training:** For a given EEG channel (e.g., `Fz`), a small neural network is trained to predict  given .
2. **Simulation:** Once trained, the network effectively "becomes" a digital replica of that brain region's signal dynamics.
3. **Stability Testing:** We inject noise into the Twin and iterate it forward.
* **Stable Brain:** The signal returns to baseline or oscillates predictably.
* **Unstable Brain:** The signal drifts significantly or diverges.


4. **Metric:** The "Variability Score" is calculated as the cumulative drift over time.

---

## Installation & Dependencies

To run this notebook, you will need a Python environment with the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn scipy torch tqdm

```

**Required Libraries:**

* `numpy` / `pandas`: Data manipulation.
* `matplotlib`: Visualization.
* `scikit-learn`: PCA and Classifiers.
* `scipy`: Signal filtering (Butterworth) and interpolation.
* `torch`: Neural networks for the Digital Twin.
* `tqdm`: Progress bars.

---

## Usage

1. **Data Preparation:**
* The data can be downloaded from Kaggle via [EEG Data](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd)
* **Update the path:** Look for the line `CSV_PATH = ...` in the second cell and update it to your local file path.
* *Note: The dataset should contain EEG columns (F3, Fz, F4, etc.) and a 'Class' column (ADHD vs Control).*


2. **Run the Notebook:**
* Execute cells sequentially. The notebook is structured to load data, process it, run classifiers, and finally train the Digital Twins.


3. **Interpret Output:**
* **Console:** Prints classification accuracy and Digital Twin training progress.
* **Plots:**
* Decision Boundaries (PCA projection).
* Brain Head Maps (Stability comparisons).





---

##  Methodology

### 1. Preprocessing

* **Sampling Rate:** 128 Hz
* **Windowing:** 4-second epochs with 50% overlap.
* **Channels:** F3, Fz, F4, C3, Cz, C4, P3, Pz, P4.

### 2. Digital Twin Architecture

A simple Feed-Forward Network is used for the twin to prevent overfitting and ensure fast convergence:

```python
nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)

```

### 3. Stability Calculation

The stability drift is calculated by auto-regressively feeding the Twin's output back into itself after a noise injection:


---

## Results

The notebook outputs visualizations comparing the **Stability Delta** between groups.

* **Blue regions** on the head map indicate higher stability (neurotypical patterns).
* **Red regions** indicate higher instability (potential markers for ADHD dynamics).

---
*This project is still being updated as I get more isntruction in computational neuroscience.*

---
*Project created for Computational Neuroscience Final Project.*
