# NEWS-Article-Summarizer
# 📰 Abstractive Text Summarization using PEGASUS

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Deep Learning project that fine-tunes Google's **PEGASUS** model to perform abstractive text summarization on the **BBC News Dataset**. This repository contains a complete pipeline from data preprocessing to a deployable web interface.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Performance Metrics](#-performance-metrics)
- [Project Structure](#-project-structure)

---

## 🔭 Overview

Abstractive summarization involves generating new sentences that capture the core meaning of the source text, rather than just selecting existing sentences (extractive). This project utilizes the **PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence)** model, which is specifically designed for this task.

We fine-tune the `google/pegasus-xsum` checkpoint on the BBC News dataset to generate concise, headline-style summaries from full news articles.

## 🌟 Key Features

* **Step-by-Step Implementation**: A comprehensive Jupyter Notebook with markdown explanations for every code block.
* **Custom Data Pipeline**: Robust data loading that handles directory traversal and text cleaning.
* **Hugging Face Integration**: Utilizes `Seq2SeqTrainer` for optimized training loops.
* **Mixed Precision Training**: Implements FP16 (floating point 16) for faster training on GPUs.
* **Comprehensive Evaluation**: Calculates ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores.
* **Interactive Web UI**: Integrated **Gradio** interface to test the model in real-time.

## 📊 Dataset

The project uses the **BBC News Summary Dataset**.
* **Source**: [Kaggle - BBC News Summary](https://www.kaggle.com/datasets/pariza/bbc-news-summary)
* **Content**: 2,225 news articles grouped into 5 categories: Business, Entertainment, Politics, Sport, and Tech.
* **Structure**: 
    * `News Articles/`: Folders containing `.txt` files of the articles.
    * `Summaries/`: Folders containing `.txt` files of the ground truth summaries.

## 🛠 Tech Stack

* **Language**: Python 3.x
* **Deep Learning**: PyTorch
* **NLP Library**: Hugging Face Transformers, Datasets, Tokenizers
* **Data Manipulation**: Pandas, Scikit-learn
* **Metrics**: ROUGE-Score, SacreBLEU
* **Deployment**: Gradio

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/yourusername/pegasus-text-summarization.git](https://github.com/yourusername/pegasus-text-summarization.git)
    cd pegasus-text-summarization
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Running the Notebook
The core logic resides in `Abstractive_Text_Summarization_BBC_News.ipynb`.

1.  Open the notebook in **Google Colab** (recommended for GPU access) or Jupyter Lab.
2.  Upload the BBC News dataset to your environment.
3.  Run the cells sequentially.

### Using the Interface
The notebook concludes with a Gradio block. Once run, it will generate a public link:

```python
import gradio as gr
# ... (Code handles prediction) ...
demo.launch(share=True)
```
## 👥 Mentorship & Support

| Role | Name |
| :--- | :--- |
| **Project Supervisor** | **Dr. Tanzila Kehkshan** |
| **Senior Mentor** | **Mr. Imran Ashraf** |

*Special thanks for their guidance, technical support, and valuable insights during the development of this project.*
