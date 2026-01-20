# DAMLE - 

# DAMLE: Domain Adaptive Meta-Learning Ensemble for Histopathology

**DAMLE** is a privacy-preserving, federated-style framework for histopathology image classification. It utilizes a **Proximity-Guided Meta-Learning** approach to synthesize predictions from frozen, hospital-specific expert models (ViT) without requiring raw data exchange or retraining large backbones.

This repository contains the implementation for training local experts on the [**Camelyon17**](https://github.com/p-lambda/wilds/tree/main/dataset_preprocessing/camelyon17) dataset and aggregating them via an Attention Gating Network.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `Prepration.ipynb` | **Step 1:** Data preprocessing, patching, and latent vector generation. Also handles t-SNE database creation and coordinate mapping. |
| `ExpertModelsCancerClassificationViT.py` | **Step 2:** Script to train individual Vision Transformer (ViT) Expert models for each hospital. |
| `DAMLE-process.ipynb` | **Step 3:** The core Meta-Learning workflow. Trains the Attention Gating Network and performs final evaluation. |

---

## 🛠️ Installation & Requirements

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/DAMLE.git
   cd DAMLE
   ```