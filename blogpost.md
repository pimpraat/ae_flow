# Study of "AE-FLOW: Autoencoders with Normalizing Flows for Anomaly Detection"

## Introduction
<!-- Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well. -->

<!-- introduce problem -->

Anomaly detection has many useful applications, such as in industrial settings and medicine, primarily for detecting abnormalities in medical images. The task of anomaly detection is complicated by the low availability of abnormal images in current datasets. For the same reason, the task of anomaly detection is often considered to be either unsupervised or semi-supervised. For several decades, anomaly detection has been an active research area (Pang et al. 2020). There is a large variety of techniques, consisting of statistics-based, distance-based, clustering-based and most recently, deep learning-based methods (Zhao et al. 2023). The deep learning based methods have consistently shown the best performance in detecting anomalies. The category of deep learning based anomaly detection consists of two primary sub-categories, namely deep learning feature extraction, and deep learning feature representations (Pang et al. 2020). The latter of which describes the investigated AE-FLOW model. Auto-encoder based models such as these are either reconstruction-based, or generative, meaning that the models learn to internally represent information in a structured manner, such that they can recreate a given input. The standard auto-encoder's capacity to model high-dimensional data distributions is however limited, leading to inaccuracies in reconstruction. An alternative category of anomaly detection methods are likelihood-based models. Likelihood-based models for anomaly detection construct a likelihood function of extracted image features. Normalizing flow (NF) is such a method, which transforms the observed image features into a versatile distribution that can be simplified. NF based methods have been shown to achieve very high performance for anomaly detection in industrial datasets. The limitation of NF based methods is that decisions are made based on the estimated likelihood of the extracted features, discarding any structural information the image may contain.

<!-- introduce solution and broad overview of methods used -->

AE-FLOW combines the autoencoder and NF anomaly detection methods by integrating a NF into the autoencoder pipeline. By doing so, Zhao et al. 2023 aim to address the limitations of each anomaly detection method. Autoencoders consist of an encoder that generates a latent representation of the input, and a decoder that reconstructs the input from the latent representation. A reconstructed image can be obtained by encoding and subsequently decoding an input. The reconstructed image fits into the distribution learned by the autoencoder, and we can compare it to the original image in order to understand if the input is out-of-distribution and therefore anomalous. The normalizing flow

AE-FLOW learns a distribution of normal images, then detects anomalies by reconstructing inputs and measuring their similarity to the original image. By learning a distribution of normal images, AE-FLOW is able to differentiate between normal images, and those with anomalies. By only utilizing normal data for training, AE-FLOW addresses the issue of limited abnormal image data availability.

<!-- -->

## Background
Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response.

Zhao, Ding, and Zhang (2023) primarily focus on the medical applications of AE-FLOW. Their findings show that AE-FLOW has very promising anomaly detection performance for medical datasets, and for one integrated circuit dataset. The evaluation into the performance of AE-FLOW on non-medical datasets is however very limited. Given the broad range of applications for anomaly detection methods and the positive results reported by Zhao, Ding, and Zhang (2023), our goal is to evaluate the performance of AE-FLOW in a broader context. 

Specifically, we aim to assess its effectiveness in the industrial domain, where anomaly detection can play a critical role in enhancing quality assurance in the manufacturing process. We do this by we training and evaluating the model on the beanTech Anomaly Detection (bTAD) dataset, which consists of real-world industrial images (cite the CEO of beans here). This will illustrate its effectiveness in novel domains, thereby providing insight into AE-FLOW's generalizability.
<!-- -->

## Method
Describe your novel contribution.




## Results
Results of your work (link that part with the code in the jupyter notebook)


## Conclusion


## Evaluation


## Contributions
* Jan Athmer - Project implementation

* Pim Praat - Project implementation

* Andre de Brandt - Writing report/blogpost

* Farrukh Baratov - Writing report/blogpost

* Thijs Wijnheijmer - Writing report/blogpost


## Bibliography


- AE-FLOW: AUTOENCODERS WITH NORMALIZING FLOWS FOR MEDICAL IMAGES ANOMALY DETECTION (Yuzhong Zhao, Qiaoqiao Ding, Xiaoqun Zhang, 2023)
- Deep Learning for Anomaly Detection: A Review (Pang et al, 2020)