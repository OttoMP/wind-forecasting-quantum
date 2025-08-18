# A quantum neural network model for short term wind speed forecasting using weather data

This repository contains the code and resources associated with our research paper: "A quantum neural network model for short term wind speed forecasting using weather data." Our work explores the application of Quantum Neural Networks (QNNs) in addressing the critical need for sustainable and accurate wind speed forecasting, a crucial component of renewable energy integration.

**Abstract:** The use of computational intelligence has become commomplace for accurate wind speed and energy forecasting, however the energy-intensive processes involved in training and tuning stands as a critical issue for the sustainability of AI models.
Quantum computing emerges as a key player in addressing this concern, offering a quantum advantage that could potentially accelerate computations or, more significantly, reduce energy consumption.
It is a matter of debate if purely quantum machine learning models, as they currently stand, are capable of competing with the classical state of the art on relevant problems. We investigate the efficacy of quantum neural networks (QNNs) for wind speed nowcasting, comparing them to a baseline Multilayer Perceptron (MLP). Utilizing meteorological data from Bahia, Brazil, we develop a QNN tailored for up to six hours ahead wind speed prediction.
Our analysis reveals that the QNN demonstrates competitive performance compared to MLP. We evaluate models using RMSE, Pearson’s R, and Factor of 2 metrics, emphasizing QNNs' promising generalization capabilities and robustness across various wind prediction scenarios.
This study is a seminal work on the potential of QNNs in advancing renewable energy forecasting, advocating for further exploration of quantum machine learning in sustainable energy research.

**Keywords:** Quantum Machine Learning, Wind speed forecasting, Renewable energy, Multivariate time series
forecasting

---
### About this Repository

We believe in the principles of open science and research transparency. Therefore, we have made all the core code and scripts used in our research openly available in this repository. Our aim is to empower fellow researchers to:

  - **Reproduce our experimental setup:** Understand the architecture and implementation of our Quantum Neural Network model.
  - **Explore Quantum Machine Learning:** Gain practical insights into applying QML for time series forecasting.
  - **Build upon our work:** Use our codebase as a foundation for future research in quantum-enhanced renewable energy forecasting.

---
### Data Availability and Reproducibility

#### Important Note on Data:

The original meteorological dataset from Bahia, Brazil, used for the experiments detailed in our research paper, contains sensitive information and cannot be publicly shared due to data access restrictions.

To ensure the reproducibility and practical utility of our code, we have provided an alternative, publicly accessible time-series weather dataset within this repository. While this public dataset is not identical to the one used in the paper's main experiments, the provided code has been adapted to demonstrate:

  - The functionality of our prediction model using a generic multivariate time-series weather dataset.
  - How to train the model to predict temperature based on various weather inputs, showcasing the adaptability of our architecture to similar forecasting tasks.

This approach allows future researchers to run the code, understand its mechanics, and adapt it for their own datasets, fostering further exploration in the field.

---
### Citation

If you find this code or research useful, please consider citing our paper:

Otto Menegasso Pires, Erick Giovani Sperandio Nascimento, and Marcelo A. Moret. A quantum neural network model for short term wind speed forecasting using weather data. Energy and AI, 21:100588, 2025.
[https://doi.org/10.1016/j.egyai.2025.100588](https://doi.org/10.1016/j.egyai.2025.100588)
