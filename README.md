# **FairVLM: Enhancing Fairness and Prompt Sensitivity in Vision–Language Models for Medical Image Segmentation**

This repository contains the official implementation of **FairVLM**, a unified framework that improves both **demographic fairness** and **prompt robustness** in vision–language models (VLMs) for medical image segmentation.

---

## Overview

Vision–Language Models (VLMs) are powerful for medical image segmentation, but their deployment faces two major issues:

- **Demographic Bias** — performance differs across demographic groups  
- **Prompt Sensitivity** — small prompt wording differences change outputs  

**FairVLM** solves both via:  
1. Semantic-Retaining Counterfactual Prompting (SRCP)  
2. Demographic-Aware Feature Normalization (DAFN)  
3. Fairness-Calibrated Loss (FCL)

---

## Key Contributions

- Counterfactual prompt generation using semantic filtering  
- Demographic-aware normalization using EMA statistics  
- Loss that penalizes group disparity & prompt inconsistency  
- Architecture-agnostic (SAMed, LViT, etc.)  
- Robust across external datasets (MosMedData+, QaTa-COV19)

---

## Method Overview

### **1. SRCP**
- Generates *m=5* diverse prompts  
- Keeps semantic similarity (cos ≥ 0.90)  
- Keeps lexical diversity (Jaccard 0.3–0.5)

### **2. DAFN**
- Computes demographic mean/std  
- Normalizes:
  \[
    \hat{z}_x = \frac{z_x - \mu_{\text{avg}}}{\sigma_{\text{avg}}}
  \]

### **3. FCL**
- Worst-case disparity penalty  
- Counterfactual prompt regularization  
- Total loss:
  \[
    \mathcal{L}_{total} = L_{base} + L_{CPR} + L_{FCL}
  \]

---

## Main Results

### Segmentation & Fairness

| Backbone | ES-Dice ↑ | DI ↓ | RPG ↓ | Robustness ↑ |
|---------|-----------|------|--------|----------------|
| SAMed + FairVLM | +1.9% | –65% | –60% | <0.5% drop |
| LViT + FairVLM  | +1.4% | –70% | –65% | <0.5% drop |

### External Generalization
Improves fairness on:
- MosMedData+  
- QaTa-COV19  

---

## Datasets
- Harvard-FairSeg  
- MosMedData+  
- QaTa-COV19  

Demographics: sex, race, ethnicity, language  

---

## Installation
````bash
git clone https://github.com/Rahman-Motiur/FairVLM.git
cd FairVLM
pip install -r requirements.txt


