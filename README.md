# Explainable Retrieval-Augmented Retinal Disease Diagnosis Using RETFound

An AI-assisted retinal disease analysis framework integrating **RETFound, Explainable Artificial Intelligence (XAI), and Retrieval-Augmented Generation (RAG)** for multi-label retinal disease prediction, visual interpretation, and supporting medical information.

---

## Overview

Retinal fundus images can contain signs of multiple diseases or abnormalities simultaneously. Identifying these conditions using automated systems is challenging, particularly when conventional classification models are designed to predict only one disease at a time.

Another important challenge is understanding how an AI model arrives at its predictions. A classification result alone does not explain which retinal regions influenced the model's decision or provide additional information about the identified condition.

This project addresses these challenges by developing a unified retinal disease analysis framework that combines three complementary components:

- **RETFound:** A retinal vision foundation model fine-tuned for multi-label retinal disease classification.
- **Explainable AI (XAI):** Generates class-specific visual attribution maps to highlight retinal regions influencing model predictions.
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant medical evidence and generates concise, disease-specific information.

The framework aims to make AI-generated retinal predictions more informative and interpretable by combining disease identification, visual explanations, and supporting medical knowledge.

## Problem Statement

To develop a retrieval-augmented retinal disease diagnosis framework using the RETFound vision foundation model for multi-label prediction of retinal diseases, Explainable AI to identify image regions influencing predictions, and Retrieval-Augmented Generation to provide relevant medical information about the predicted conditions.

## Project Objectives

- Develop a retinal disease analysis framework using the RETFound vision foundation model.
- Perform multi-label classification across ten selected retinal conditions.
- Apply Explainable AI techniques to visualize the retinal regions influencing individual predictions.
- Integrate Retrieval-Augmented Generation to retrieve relevant medical information for predicted conditions.
- Combine disease predictions, visual explanations, and supporting medical knowledge into a unified system.
- Evaluate the classification model using Precision, Recall, F1-score, AUC, and Exact Match.

## Dataset Preparation

### Original Dataset

The project uses the **Retinal Fundus Multi-Disease Image Dataset (RFMiD)** as its original data source.

The original RFMiD dataset contains 3,200 retinal fundus images annotated with multiple disease and pathology categories. It supports multi-label retinal image analysis, where an image can contain more than one condition.

### Modified RFMiD Dataset

A modified dataset was prepared specifically for this project by selecting ten target classes: nine retinal disease/pathology categories and one Normal class.

The selected conditions are:

| No. | Class | Description |
|---:|---|---|
| 1 | ARMD | Age-related Macular Degeneration |
| 2 | BRVO | Branch Retinal Vein Occlusion |
| 3 | DN | Diabetic Neuropathy |
| 4 | DR | Diabetic Retinopathy |
| 5 | MH | Macular Hole |
| 6 | MYA | Myopia |
| 7 | Normal | Normal |
| 8 | ODC | Optic Disc Cupping |
| 9 | ODP | Optic Disc Pallor |
| 10 | TSLN | Tessellation |

### Dataset Selection and Processing

- Selected nine retinal disease/pathology categories along with Normal.
- Retained images based on the condition that the number of selected disease labels was greater than or equal to the number of ignored disease labels.
- Preserved the multi-label nature of the selected annotations.
- Organized the resulting images into training, validation, and testing sets.
- Applied multi-label stratification to maintain a suitable distribution of labels across the splits.

### Final Dataset Distribution

| Dataset Split | Images |
|---|---:|
| Training | 1,906 |
| Validation | 402 |
| Testing | 402 |
| **Total** | **2,710** |

An image can contain multiple positive disease labels, allowing the model to identify more than one retinal condition from a single fundus image.

### Dataset and Model Resources

The following resources were prepared and published on Kaggle as part of this project:

| Resource | Description | Link |
|---|---|---|
| Modified RFMiD 10 Diseases | Project-prepared 10-class retinal fundus dataset | [View Dataset](https://www.kaggle.com/datasets/krishba/retfound-rfmid-10-diseases) |
| RETFound Best Model Checkpoint | Fine-tuned model checkpoint selected using validation performance | [View Checkpoint](https://www.kaggle.com/datasets/rithika2006/retfound-best-model-checkpoint) |

## Methodology

The proposed framework consists of three major components: RETFound-based classification, Explainable AI, and Retrieval-Augmented Generation.

### 1. RETFound-Based Multi-Label Classification

RETFound is a retinal vision foundation model pretrained on large-scale retinal images using self-supervised learning.

In this project, the pretrained RETFound model was fine-tuned to perform multi-label classification across the ten selected retinal conditions.

**Training approach:**

- Adapted RETFound for ten-class multi-label classification.
- Used `BCEWithLogitsLoss` with class weights to address class imbalance.
- Used the AdamW optimizer.
- Applied mixed-precision training.
- Monitored validation performance to select the best model checkpoint.
- Optimized class-specific decision thresholds using the validation set.

**Training configuration:**

| Parameter | Configuration |
|---|---|
| Base Model | RETFound |
| Classification Type | Multi-label |
| Number of Classes | 10 |
| Loss Function | BCEWithLogitsLoss with class weights |
| Optimizer | AdamW |
| Learning Rate | 5 × 10⁻⁵ |
| Weight Decay | 0.05 |
| Batch Size | 8 |
| Maximum Epochs | 30 |
| Training Precision | Mixed Precision |
| Computational Hardware | 2 × NVIDIA Tesla T4 GPUs |
| Best Checkpoint | Epoch 12 |
| Best Validation Macro-F1 | 0.7009 |

After training, class-specific thresholds were optimized instead of applying a single fixed threshold to all classes. These thresholds were used to obtain the final multi-label predictions.

### 2. Explainable Artificial Intelligence (XAI)

The XAI component was developed to provide visual explanations for the predictions generated by RETFound.

Rather than presenting only disease probabilities, the system generates class-specific attribution maps to identify retinal regions that influenced individual predictions.

**XAI approach:**

- Extracted gradient information corresponding to individual predicted classes.
- Used Transformer patch activations from the RETFound model.
- Combined gradients and patch-level activations to calculate attribution scores.
- Generated class-specific heatmaps.
- Overlaid the heatmaps on the original retinal fundus images.

The resulting visualizations help examine which regions contributed to the model's predictions, including cases where multiple conditions were predicted for the same image.

**Faithfulness Evaluation:**

A perturbation-based experiment was conducted using 52 image-condition pairs from the test predictions.

Three patch-selection strategies were compared:

- Highest-attribution patches.
- Randomly selected patches.
- Lowest-attribution patches.

The experiment was performed at five masking levels: 5%, 10%, 20%, 30%, and 40%.

The results showed that masking highly attributed regions produced a greater reduction in the target-class prediction probability than masking random or low-attribution regions.

This provides evidence of model-level faithfulness of the generated explanations.

### 3. Retrieval-Augmented Generation (RAG)

The RAG component provides additional medical information related to the conditions predicted by RETFound.

A curated medical knowledge corpus was prepared to serve as the information source for disease-specific retrieval and response generation.

**RAG workflow:**

1. Receive the retinal disease predictions generated by RETFound.
2. Generate disease-specific retrieval queries based on the predicted conditions.
3. Convert the queries and medical evidence into semantic embeddings.
4. Retrieve relevant medical information from the knowledge corpus.
5. Use the retrieved evidence as contextual input to a language model.
6. Generate concise, condition-specific medical information.

The generated information may include:

- Disease descriptions and definitions.
- Causes and associated factors.
- Symptoms and clinical characteristics.
- Treatment-related information available in the retrieved evidence.

**Medical Knowledge Base Configuration:**

| Parameter | Configuration |
|---|---|
| Retinal Conditions Covered | 10 |
| Verified Medical Evidence Entries | 63 |
| Embedding Dimension | 384 |
| Vector Index | FAISS IndexFlatIP |
| Report Sections | 4 |
| Disease Coverage | 10/10 |
| Section Coverage | 4/4 |

The retrieval process is performed independently for each predicted condition. When multiple diseases are predicted, the system generates separate supporting information for each condition.

### 4. Overall System Integration

The three components are integrated into a unified workflow.

**Input:** Retinal fundus image.

**Processing:**

1. RETFound analyzes the image and generates probabilities for the ten target classes.
2. Optimized class-specific thresholds are applied to obtain the final multi-label predictions.
3. The XAI module generates class-specific visual explanations for the predicted conditions.
4. The RAG module retrieves relevant medical evidence and generates condition-specific information.

**Final Output:**

- Predicted retinal conditions and their probabilities.
- Class-specific XAI heatmaps and visual overlays.
- Retrieved medical evidence and generated medical summaries.

The integrated system provides three complementary forms of information:

| Component | Main Output |
|---|---|
| RETFound | What conditions are predicted? |
| XAI | Which image regions influenced the prediction? |
| RAG | What relevant medical information is available about the predicted conditions? |

## Results and Performance

### RETFound Classification Results

The fine-tuned RETFound model achieved the following performance on the 402-image test set:

| Evaluation Metric | Score |
|---|---:|
| Macro F1-Score | 0.7008 |
| Micro F1-Score | 0.7636 |
| Macro Precision | 0.7179 |
| Macro Recall | 0.7012 |
| Macro AUC | 0.9216 |
| Exact Match Ratio | 0.6418 |

The best validation checkpoint was obtained at Epoch 12, with a validation Macro-F1 score of 0.7009.

### XAI Results

- Generated class-specific heatmaps for predicted retinal conditions.
- Produced visual overlays on the original fundus images.
- Supported explanations for both single-label and multi-label predictions.
- Evaluated attribution faithfulness through a perturbation-based experiment.

### RAG Results

- Prepared a medical evidence corpus covering all ten target conditions.
- Retrieved condition-specific medical information using semantic embeddings.
- Used retrieved evidence as context for language-model response generation.
- Generated separate medical summaries for multiple predicted conditions.

### Integrated System Results

The final framework combines multi-label retinal disease prediction, visual interpretability, and supporting medical information into a single analysis workflow.

## Kaggle Project Resources

The following Kaggle resources were created and published as part of this project. They contain the prepared datasets, model checkpoint, and generated outputs.

| Resource | Contents | Link |
|---|---|---|
| Modified RFMiD 10 Diseases | Modified retinal image dataset with ten target classes | [Open on Kaggle](https://www.kaggle.com/datasets/krishba/retfound-rfmid-10-diseases) |
| RETFound Best Model Checkpoint | Fine-tuned RETFound model checkpoint | [Open on Kaggle](https://www.kaggle.com/datasets/rithika2006/retfound-best-model-checkpoint) |
| Retina - RETFound Final Results | Classification metrics, optimized thresholds, class-wise results, retrieval evidence, and generated summaries | [Open on Kaggle](https://www.kaggle.com/datasets/rithika2006/project-phase-1-retfound-results) |
| Clean Medical Evidence Corpus | Curated medical knowledge corpus used by the RAG component | [Open on Kaggle](https://www.kaggle.com/datasets/rithika2006/clean-medical-evidence-corpus) |
| Retina - XAI Final Results | XAI result records, evaluation summaries, and visualizations for test images | [Open on Kaggle](https://www.kaggle.com/datasets/rithika2006/project-phase-1-xai-final-results) |

These resources are maintained separately from the GitHub notebook repository to keep the repository focused on implementation and documentation while providing access to the larger datasets and generated outputs.

## Repository Structure

This GitHub repository contains the Jupyter notebooks used to develop, test, and integrate the project components.

| Notebook | Purpose |
|---|---|
| `Dataset preprocessing` | Preparation and preprocessing of the modified RFMiD dataset |
| `RETFound Training` | Fine-tuning and evaluation of the RETFound model |
| `XAI Component of Retinal project` | Implementation of the XAI component |
| `XAI - Inspect` | Inspection and visualization of XAI outputs |
| `XAI Verification Retina` | Verification and evaluation of XAI explanations |
| `RAG_Comp_medical_corpus` | Preparation of the medical evidence corpus and RAG component |
| `RAG Verification Retina` | Verification of medical evidence retrieval and generated information |
| `Retina Detection + visual reference-demo` | Retinal detection and visual reference implementation |
| `Final RAG + XAI + prediction demo` | Integration of prediction, XAI, and RAG components |

## Technologies and Tools

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Vision Foundation Model | RETFound |
| Machine Learning | Scikit-learn |
| Data Processing | NumPy, Pandas |
| Explainable AI | Gradient-based attribution, Transformer patch activations |
| Vector Search | FAISS |
| Semantic Embeddings | Sentence embedding model |
| Generative AI | Large Language Model (LLM) |
| Visualization | Matplotlib |
| Development Environment | Kaggle Notebooks |
| Hardware | NVIDIA Tesla T4 GPUs |

## How to Explore the Project

1. Clone or download this GitHub repository.
2. Open the notebooks in Jupyter Notebook or Kaggle.
3. Follow the dataset preprocessing notebook to understand the data preparation process.
4. Explore the RETFound training notebook to understand model fine-tuning and evaluation.
5. Review the XAI notebooks to examine visual explanations.
6. Explore the RAG notebooks to understand medical evidence retrieval and response generation.
7. Refer to the integrated demo notebook to understand how all three components work together.

For the modified dataset, trained checkpoint, and generated outputs, refer to the Kaggle resources linked above.

## References

1. Pachade, S., Porwal, P., Thulkar, D., Kokare, M., Deshmukh, G., Sahasrabuddhe, V., Giancardo, L., Quellec, G., & Mériaudeau, F. (2021). Retinal Fundus Multi-Disease Image Dataset (RFMiD): A dataset for multi-disease detection research. *Data, 6*(2), 14. https://doi.org/10.3390/data6020014

2. Zhou, Y., Chia, M. A., Wagner, S. K., Ayhan, M. S., Williamson, D. J., Struyven, R. R., Liu, T., Xu, M., Lozano, M. G., Woodward-Court, P., Kihara, Y., UK Biobank Eye & Vision Consortium, Altmann, A., Lee, A. Y., Topol, E. J., Denniston, A. K., Alexander, D. C., & Keane, P. A. (2023). A foundation model for generalizable disease detection from retinal images. *Nature, 622*, 156–163. https://doi.org/10.1038/s41586-023-06555-x

## Disclaimer

This project is an academic research prototype developed for retinal image analysis and medical information retrieval. The model predictions, visual explanations, and generated medical information are intended to provide additional context and support understanding. They are not a substitute for professional ophthalmological examination, clinical diagnosis, or medical advice.
