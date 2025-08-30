# 🤖 Machine Learning Project: Work Category Classifier 🚀
**🎯 Prediction of the work category based on the incidence description**
**🇪🇸 Spanish Language NLP Model**

---

## 📋 Table of Contents
1. #### [🔍 Project Overview](#-project-overview)
    - [🎯 Goal](#-goal)
    - [❓ Problem Statement](#-problem-statement)
    - [📊 Metrics](#-metrics)
    - [🛠️ Approach](#️-approach)
2. #### [📁 Data](#-data)
    - [📥 Data Collection](#-data-collection)
    - [🔬 Data Exploration (EDA)](#-data-exploration-eda)
3. #### [⚙️ Methodology](#️-methodology)
    - [🧹 Data Preprocessing](#-data-preprocessing)
    - [🔧 Feature Engineering](#-feature-engineering)
    - [🎲 Model Selection](#-model-selection)
    - [🏋️ Model Training](#️-model-training)
    - [📈 Model Evaluation](#-model-evaluation)
    - [🚀 Model Deployment](#-model-deployment)
4. #### [📊 Results](#-results)
5. #### [🎯 Conclusion & Future Work](#-conclusion--future-work)
6. #### [💾 Installation](#-installation)
7. #### [🖥️ Usage](#️-usage)

---

## 🔍 Project Overview

### 🎯 Goal
To predict technician work category using **Spanish NLP**, improving repairing/maintenance response time, eliminating workload and avoiding non-job-related learning for Spanish-speaking hospital staff.

### ❓ Problem Statement
Workers in the hospital send a repairing/maintenance request through a ticketing software. They have to specify the category of the maintenance technician to whom is intended (⚡ electrician, 🔧 mechanic, 🚰 plumber, 🪚 carpenter, etc.) but often the incidences are miscategorized because the healthcare personnel hasn't been trained for it. This situation leads to a slower response of the maintenance team and extra workload (and frustration) to the healthcare personnel that has to learn constantly who is in charge of every part of the equipment and building instead of performing their duties. The situation is compounded when new short term employees arrive.

### 📊 Metrics
Cause the number of incidences is different for every category (imbalanced classes), I opt for **Precision**, **Recall**, and **F1-Score** with Macro Average, so all the categories are treated equally. Confusion matrix is used for exploring.

### 🛠️ Approach
Starting collecting the data in the ticketing software. The field of interest are processed and prepared for training. Different model topologies are tested in the search for the best one. The final model is tested and evaluated, putting the focus on every class individually. Finally the model is deployed in HuggingFace and future improvements are discussed.

---

## 📁 Data

### 📥 Data Collection
All the data is provided by the ticketing software in **CSV format**. All the sensitive data is filtered by inside the software request and just "Observation" and "Category" fields are used in this project. **The dataset contains incident descriptions written in Spanish** by hospital staff.

### 🔬 [Data Exploration (EDA)](notebooks/1_data_exploration.ipynb)
In this stage it's visualized how the data is stored in the csv file and decide what can and cannot be useful for the model. This will help us decide how to proceed in the subsequent stages.

![📊 Word Cloud of vocabulary](notebooks/src/img/exploration.png)

---

## ⚙️ Methodology

### 🧹 [Data Preprocessing](notebooks/2_data_preprocessing.ipynb)
Prepare the data to be fed to the training stage, eliminating outliers, grouping small classes, cleaning the data if necessary, split into training and test dataset, vectorization and creating batches.

![📈 Boxplot text size per class](notebooks/src/img/preprocessing.png)

### 🎲 [Model Selection & Training](notebooks/3_model_training.ipynb)
To find the best model topology to predict between **12 different classes**. Starting with the most basic DNN until RNNs and CNNs, several models are tested and fine-tuned using validation results to choose the best performing model.

![🎯 Validation performance per class](notebooks/src/img/training.png)

### 📈 [Model Evaluation](notebooks/4_model_evaluation.ipynb)
To determine how the model performs to unseen data, putting attention to every specific class. Using **Confusion Matrix**, **Error Analysis** and **Regression** each class is tested and examined looking for anomalies in predictions, explaining the reason for the majority of misclassification. Finally it's proven how the size of each class in the dataset affects the model forecast.

![📊 Size dataset and accuracy correlation](notebooks/src/img/evaluation.png)

### 🚀 [Model Deployment](https://huggingface.co/spaces/david-delpico-ml/ticketing)
The final model is deployed using **Gradio** in [🤗 HuggingFace](https://huggingface.co/spaces/david-delpico-ml/ticketing)

![🖥️ Screenshot HuggingFace interface](notebooks/src/img/huggingface.png)

---

## 📊 Results

The final forecast of the model can be marked as a **✅ success**. The accuracy of the model is established at **85%**, reaching F1-Score of **89%** as highest in some classes and **0.09** the lowest. Some work in the model input and in the dataset should be necessary to sort the imprecision in three of the 12 classes. Lack of entries in certain categories and mixed work related fields are the reasons for the underperformance of those classes. The model could be used and implemented because exceeds the **80% accuracy** threshold so can be stated that the model learned.

---

## 🎯 Conclusion & Future Work

- ✅ Reaching an overall performance accuracy of **85%**, the resulting model can outperform the actual manual system where **20%** of the tickets are incorrectly labeled.
- 🔊 The model can be implemented along with a **TextToSpeech** system for consulting or call forwarding to the relevant team.
- 🎯 Create a **Multi-modal input system** to accept other available fields like location to be more accurate in the prediction.
- 🤖 Use **attention mechanism** within transformers.
- 🏷️ Allow **multi-classification** because sometimes more than one team is related to a ticket.

---

## 💾 Installation

```bash
# Clone the repository
git clone https://github.com/david-delpico-ml/ticketing-classifier.git

# Install dependencies
pip install -r requirements.txt
```

## 🖥️ Usage

```python
# Import the model
from src.features.predict import predict

# Initialize and predict (Spanish text input)
prediction = predict("La luz del quirófano no funciona correctamente")
print(f"🎯 Predicted category: {prediction}")
```

---

<div align="center">

### 🌟 **Made with ❤️ for better hospital maintenance workflows** 🌟

**⭐ Star this repo if you found it helpful!**

</div>