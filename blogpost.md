# Study of "AE-FLOW: Autoencoders with Normalizing Flows for Anomaly Detection"

## Introduction
<!-- Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. 


TODO: It should contain one paragraph of related work as well. 
TODO: Limitations
TODO: Add citations


-->

<!-- introduce problem -->
Anomaly detection has many useful applications, such as in industrial settings and medicine, primarily for detecting abnormalities in medical images. The task of anomaly detection is complicated by the low availability of abnormal images in current datasets. For the same reason, the task of anomaly detection is often considered to be either unsupervised or semi-supervised. For several decades, anomaly detection has been an active research area (Pang et al. 2020). There is a large variety of techniques, consisting of statistics-based, distance-based, clustering-based and most recently, deep learning-based methods (Zhao et al. 2023). The deep learning based methods have consistently shown the best performance in detecting anomalies. The category of deep learning based anomaly detection consists of two primary sub-categories, namely deep learning feature extraction, and deep learning feature representations (Pang et al. 2020). The latter of which describes the investigated AE-FLOW model. Auto-encoder based models such as these are reconstruction-based, meaning that the models learn to internally represent information in a structured manner, such that they can recreate a given input. 

An autoencoder (AE) consists of an encoder that generates the representation, and the decoder that reconstructs the input from the latent representation. Autoencoders are often used in Generative Adversarial Networks (GAN), a model that consists of a traditional AE and an additional subnetwork known as a discriminator. GANs' key assumption is that if the AE learns the right features, then its decoder can be fed randomly generated inputs and produce outputs that appear to be drawn from the input distribution. Therefore, we can better train the AE by also training a discriminator to distinguish between real and fake inputs. This means the decoder has to learn to generate more convincing outputs, while the discriminator has to get better at telling them apart. One example is AnoGAN, a model that uses GANs and with a specialized weighted sum of the residual and discrimination score in order to detect anomalies. Another method, GANomaly detects anomalies based on the distance of inputs in the latent feature space. However, tthe standard auto-encoder's capacity to model high-dimensional data distributions is limited, which leads to inaccuracies in reconstruction. 

Likelihood-based anomaly detection models present an alternative approach. They work by constructing a likelihood function of extracted image features. One such method is the normalizing flow (NF), which transforms an observed image's features into a versatile distribution that can be simplified. This is done by utilizing a using a sequence of invertible transformations that simplify complex distributions. NF based methods have been shown to achieve very high performance for anomaly detection in industrial datasets. 
One such method is Differnet, which utilized a normalizing flow subnetwork to maximize likelihood. <!-- Maybe still discuss fastflow-->
The limitation of NF based methods is that decisions are made based on the estimated likelihood of the extracted features, discarding any structural information the image may contain.

<!-- Related work (one paragraph) TODO: 
AnoGAN -> anomaly score

f-AnoGAN
GANomaly
DifferNet
Fastflow

-->





<!-- introduce solution and broad overview of methods used -->

AE-FLOW combines the autoencoder and NF anomaly detection methods by integrating a NF into the autoencoder pipeline. By doing so, Zhao et al. 2023 aim to address the limitations of each anomaly detection method. Autoencoders consist of an encoder that generates a latent representation of the input, and a decoder that reconstructs the input from the latent representation. A reconstructed image can be obtained by encoding and subsequently decoding an input. The reconstructed image fits into the distribution learned by the autoencoder, and we can compare it to the original image in order to understand if the input is out-of-distribution and therefore anomalous. The normalizing flow

AE-FLOW learns a distribution of normal images, then detects anomalies by reconstructing inputs and measuring their similarity to the original image. By learning a distribution of normal images, AE-FLOW is able to differentiate between normal images, and those with anomalies. By only utilizing normal data for training, AE-FLOW addresses the issue of limited abnormal image data availability.


<!-- Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response. -->
AE-FLOW's results prove it to be a promising approach to anomaly detection, with good performance on multiple medical datasets and one integrated circuit dataset. Furthermore, its approach of combining the two already established methods of anomaly detection is simple yet elegant. However, our results so far are limited to a few datasets and two fields. We need to evaluate AE-FLOW's performance with datasets from new domains in order to understand how well it is able to perform in different domains. 


<!-- Describe your novel contribution. -->
We have two main contributions. First, we reimplement and evaluate the AE-FLOW model in Python, which we do by closely following the specifications outlined in Section 3.2. These specifications specify the exact number of layers, layer types, and layer sizes. This allows us to reimplement the encoder, decoder, and normalizing flow with close fidelity to what we expect the authors to have originally used. The repository containing the authors' implementation is not publicly available, therefore this contribution is necessary for to reproduce the author's results.

Our second contribution is testing AE-FLOW's generalizability. Zhao, Ding, and Zhang (2023) primarily focus on the medical applications of AE-FLOW. Their findings show that AE-FLOW has very promising anomaly detection performance for medical datasets, and for one integrated circuit dataset. The evaluation into the performance of AE-FLOW on non-medical datasets is however very limited. Given the broad range of applications for anomaly detection methods and the positive results reported by Zhao, Ding, and Zhang (2023), our goal is to evaluate the performance of AE-FLOW in a broader context. 

Specifically, we aim to assess its effectiveness in the industrial domain, where anomaly detection can play a critical role in enhancing quality assurance in the manufacturing process. We do this by we training and evaluating the model on the beanTech Anomaly Detection (bTAD) dataset, which consists of real-world industrial images (Mishra et al., 2021). This will illustrate its effectiveness in novel domains, thereby providing insight into AE-FLOW's generalizability.



## Results
<!-- Results of your work (link that part with the code in the jupyter notebook) -->

<!--
results on xray-dataset
- using the resnet-like subnet very similar performance (when using resnet like subnet):
- ours: {'AUC': 0.8196581196581196, 'ACC': 0.8589743589743589, 'SEN': 0.9769230769230769, 'SPE': 0.6623931623931624, 'F1': 0.8964705882352941}
- theirs: AUC: 0.92, F1: 0.88, ACC: 0.85, SEN: 0.91, SPE: 0.76

ours on convnet like subnet
F1: 0.764, ACC: 0.6522, SEN: 0.9231, SPE: 0.2009, AUC: 0.562

 -->
Our results show performance similar to that of Zhao et al. on the chest-XRAY dataset when using the ResNet type subnet for the normalizing flow submodule. The utilization of the alternate subnet, which comprises two convolutional layers and a ReLU activation function, resulted in a notable decrease in performance. This was evidenced by the F1-score, which ended up being 20% lower.



## Contributions
* Jan Athmer - Project implementation

* Pim Praat - Project implementation

* Andre de Brandt - Writing report/blogpost, debugging

* Farrukh Baratov - Writing report/blogpost, debugging

* Thijs Wijnheijmer - Writing report/blogpost, debugging


## Bibliography


- AE-FLOW: AUTOENCODERS WITH NORMALIZING FLOWS FOR MEDICAL IMAGES ANOMALY DETECTION (Yuzhong Zhao, Qiaoqiao Ding, Xiaoqun Zhang, 2023)
- Deep Learning for Anomaly Detection: A Review (Pang et al, 2020)
- VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization (Pankaj Mishra and Riccardo Verk and Daniele Fornasier and Claudio Piciarelli and Gian Luca Foresti, 2021)