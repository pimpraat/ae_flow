# Study of "AE-FLOW: Autoencoders with Normalizing Flows for Anomaly Detection"

## Introduction
<!-- Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well. -->

<!-- introduce problem -->

Anomaly detection has many useful applications, such as in industrial environments and medicine. In medicine, the predominant applications are in detecting abnormalities in medical images. A major challenge in anomaly detection is the low availability of abnormal images in current datasets. The task of anomaly detection has been an active area of research for decades (Pang et al. 2020) and is often considered non-supervised or semi-supervised. There exists a wide variety of techniques applied in anomaly detection consisincluding: of statistics-based, distance-based, clustering-based and recently deep learning-based methods (Zhao et al. 2023). 

The deep learning based methods have consistently shown the best performance in detecting anomalies. This deep learning-based category for anomaly detection consists of two primary subcategories, namely deep learning feature extraction and deep learning feature representations (Pang et al. 2020). The latter describes the AE-FLOW model studied. Auto-encoder-based models such as these are either reconstruction-based or generative. Generative means that the models learn to represent information internally in a structured way, so that they can recreate a given input. However, the ability of the standard auto-encoder to model high-dimensional data distributions is limited, leading to reconstruction inaccuracies. 

An alternative category of anomaly detection methods are likelihood-based models. Likelihood-based models for anomaly detection construct a likelihood function of extracted image features. Normalising Flow (NF) is such method, which converts the observed image features into a versatile distribution that can be simplified. Methods based on NF are found to have very good performance for anomaly detection in industrial datasets. The limitation of NF-based methods is the fact that decisions are made based on the estimated probabilities of the extracted features. This disregards any structural information that the image may contain.

<!-- introduce solution and broad overview of methods used -->

AE-FLOW combines the anomaly detection methods autoencoder and NF by integrating an NF into the autoencoder pipeline. In doing so, Zhao et al 2023 aim to address the limitations of anomaly detection methods. Autoencoders consist of an encoder that generates a latent representation of the input, and a decoder that reconstructs the input from the latent representation. A reconstructed image can be obtained by encoding and then decoding an input. The reconstructed image fits the distribution learned by the auto-encoder, and we can compare it with the original image to understand whether the input falls outside the distribution and is therefore anomalous.

<!-- The normalizing flow... ? this part was not finished yet -->

AAE-FLOW learns a distribution of normal images and then detects anomalies by reconstructing the input and measuring its similarity to the original image. By learning a distribution of normal images, AE-FLOW can distinguish between normal images and images with anomalies. By only utilizing normal data for training, AE-FLOW addresses the issue of limited abnormal image data availability.

<!-- -->

## Background
<!-- Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response. -->

Zhao, Ding, and Zhang (2023) primarily focus on the medical applications of AE-FLOW. Their findings show that AE-FLOW has very promising anomaly detection performance for medical datasets, and for one integrated circuit dataset. The evaluation into the performance of AE-FLOW on non-medical datasets is however very limited. Given the broad range of applications for anomaly detection methods and the positive results reported by Zhao, Ding, and Zhang (2023), our goal is to evaluate the performance of AE-FLOW in a broader context. 

Specifically, we aim to assess its effectiveness in the industrial domain, where anomaly detection can play a critical role in enhancing quality assurance in the manufacturing process. We do this by we training and evaluating the model on the beanTech Anomaly Detection (bTAD) dataset, which consists of real-world industrial images (Mishra et al., 2021). This will illustrate its effectiveness in novel domains, thereby providing insight into AE-FLOW's generalizability.
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
- "VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization" (P. Mishra, R. Verk, D. Fornasier, C. Piciarelli, G.L. Foresti, 2021)
