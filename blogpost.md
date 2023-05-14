# Study of "AE-FLOW: Autoencoders with Normalizing Flows for Anomaly Detection"

## Introduction
<!-- Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. 
-->

<!--  beginning + related work -->
Anomaly detection has many useful applications, such as in industrial settings and medicine, primarily for detecting abnormalities in medical images. The task of anomaly detection is complicated by the low availability of abnormal images in current datasets. For the same reason, the task of anomaly detection is often considered to be either unsupervised or semi-supervised. For several decades, anomaly detection has been an active research area (Pang et al. 2020). There is a large variety of techniques, consisting of statistics-based, distance-based, clustering-based and most recently, deep learning-based methods (Zhao et al. 2023). The deep learning based methods have consistently shown the best performance in detecting anomalies. The category of deep learning based anomaly detection consists of two primary sub-categories, namely deep learning feature extraction, and deep learning feature representations (Pang et al. 2020). The latter of which describes the investigated AE-FLOW model. Auto-encoder based models such as these are reconstruction-based, meaning that the models learn to internally represent information in a structured manner, such that they can recreate a given input. 

An autoencoder (AE) consists of an encoder that generates the representation, and the decoder that reconstructs the input from the latent representation. Autoencoders are often used in Generative Adversarial Networks (GAN), a model that consists of a traditional AE and an additional subnetwork known as a discriminator. GANs' key assumption is that if the AE learns the right features, then its decoder can be fed randomly generated inputs and produce outputs that appear to be drawn from the input distribution. Therefore, we can better train the AE by also training a discriminator to distinguish between real and fake inputs. This means the decoder has to learn to generate more convincing outputs, while the discriminator has to get better at telling them apart. One example is AnoGAN, a model that uses GANs and with a specialized weighted sum of the residual and discrimination score in order to detect anomalies. Another method named 'GANomaly' detects anomalies based on the distance of inputs in the latent feature space. However, the standard auto-encoder's capacity to model high-dimensional data distributions is limited, which leads to inaccuracies in reconstruction. 

Likelihood-based anomaly detection models present an alternative approach. They work by constructing a likelihood function of extracted image features. One such method is the normalizing flow (NF), which transforms an observed image's features into a versatile distribution that can be simplified. This is done by using a sequence of invertible transformations that simplify complex distributions. NF based methods have been shown to achieve very high performance for anomaly detection in industrial datasets. 
One such method is Differnet, which utilized a normalizing flow subnetwork to maximize likelihood. <!-- Maybe still discuss fastflow-->
The limitation of NF based methods is that decisions are made based on the estimated likelihood of the extracted features, discarding any structural information the image may contain.

<!-- An analysis of the paper and its key components. -->

AE-FLOW combines the autoencoder and NF anomaly detection methods by integrating a NF into the autoencoder pipeline. By doing so, Zhao et al. 2023 aimed to address the limitations of each anomaly detection method. Autoencoders consist of an encoder that generates a latent representation of the input, and a decoder that reconstructs the input from the latent representation. A reconstructed image can be obtained by encoding and subsequently decoding an input. The reconstructed image fits into the distribution learned by the autoencoder, and we can compare it to the original image in order to understand if the input is out-of-distribution and therefore anomalous.
AE-FLOW learns a distribution of normal images, then detects anomalies by reconstructing inputs and measuring their similarity to the original image. By learning a distribution of normal images, AE-FLOW is able to differentiate between normal images, and those with anomalies. By only utilizing normal data for training, AE-FLOW addresses the issue of limited abnormal image data availability.


The architecture of AE-FLOW is pictured below. It resembles a standard autoencoder, however the latent representation of the input _z_ is put through a normalizing flow to produce the normalized representation _z'_. The normalizing flows are trained with normal data to ensure that abnormal features are poorly captured in the learned distribution. This prevents the decoder from effectively reconstructing abnormal features, which enables us to detect anomalies based on the difference between the original, potentially abnormal, image and the reconstructed and normalized image.

<!-- ADD A FIGURE WITH THE AE-FLOW ARCHITECTURE HERE -->

The  is traiend using a loss function that accounts for reconstruction accuracy at the pixel level and the distribution likelihood at the feature level. This way, it ensures that the different components are trained to perform well at the intended frame of reference. It utilizes a corresponding anomaly score function comprised of the reconstruction error and the flow likelihood to detect out-of-distribution data. 

AE-FLOW's novel approach proves useful, exhibiting significant improvements with multiple metrics across all tested datasets. It is most consistent with the ISIC 2018 dataset (Codella et al. 2018), where it outperforms all five tested reconstruction and likelihood-based models across five different metrics, exhibiting percentage increases of up to 40.1 points. 

<!-- Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response. -->
The results prove it to be a promising approach to anomaly detection, with good performance on multiple medical datasets and one integrated circuit dataset. This shows that AE-FLOW can potentially be suited to perform tasks across all domains, rather than simply being a one-trick medical pony. Furthermore, its approach of combining the two previously established methods of anomaly detection is simple yet elegant. It combines two different architectures such that they compensate for each others weaknesses while avoiding the additional complications that arise when completely novel and untested architectures are introduced. Overall, AE-FLOW is an architecture that is worth investigating further.


However, this architecture is quite new, and therefore we know very little about domains and applications in which it excels. We have evidence of good results in the medical domain, and some insight into its capabilities with integrated circuits. However, we have no results that prove its effectiveness in other domains, such as industrial applications. AE-FLOW needs more extensive training with datasets from other domains if we are to properly understand its strengths and limitations as an established method of anomaly detection.


<!-- Describe your novel contribution. -->
We have two main contributions. First, we implement and evaluate the AE-FLOW model in Python, which we do by closely following the specifications outlined in Section 3.2. These specifications specify the exact number of layers, as well as their types and sizes. This allows us to reimplement the encoder, decoder, and normalizing flow with close fidelity to what we expect the authors to have originally used. The repository containing the authors' implementation is not publicly available, therefore this contribution 
is necessary for to reproduce the author's results. We further seek to replicate the results in the paper to ensure that our implementation is faithful to the authors' model.

Our second contribution is testing AE-FLOW's generalizability. The authors primarily focussed on the medical applications of AE-FLOW. Their findings showed that AE-FLOW has very promising anomaly detection performance for medical datasets, and for one integrated circuit dataset. The evaluation into the performance of AE-FLOW on non-medical datasets is however very limited. Given the broad range of applications for anomaly detection methods and the positive results reported by Zhao, Ding, and Zhang (2023), our goal is to evaluate the performance of AE-FLOW in a broader context. Specifically, we aim to assess its effectiveness in the industrial domain, where anomaly detection can play a critical role in enhancing quality assurance in the manufacturing process. We do this b training and evaluating the model on the beanTech Anomaly Detection (bTAD) dataset, which consists of real-world industrial images (Mishra et al., 2021). This will illustrate its effectiveness in novel domains, thereby providing insight into AE-FLOW's generalizability.

We chose the bTAD dataset because it is publicly available and contains industrial images, which we would expect AE-FLOW to perform well on given the track record of similar models. If AE-FLOW performs as well as expected, then we can also be certain that it has been able to get good results in a domain that anomaly detection models performed worse at before (medical domains) while still performing well in familiar domains (industrial domains).


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

|  | dataset | subnet | F1-score | ACC | AUC | SEN | SPE | 
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| AE-FLOW (Zhao et al.) | chest-XRAY | ResNet-Like | 0.88 | 0.85 | 0.92 | 0.91 | 0.76 |
| AE-FLOW (Ours) | chest-XRAY | ResNet-Like | 0.89 | 0.86 | 0.82 | 0.98 | 0.66 |
| AE-FLOW (Ours) | chest-XRAY | ConvNet-Like | 0.76 | 0.65 | 0.56 | 0.92 | 0.20 |
| AE-FLOW (Zhao et al.) | OCT2017 | ConvNet-Like | 0.96 | 0.94 | 0.98 | 0.97 | 0.88 |
| AE-FLOW (Ours) | OCT2017 | ConvNet-Like | 0.75 | 0.61 | 0.43 | 0.78 | 0.08 |
| AE-FLOW (Ours) | OCT2017 | ResNet-Like | 0.71 | 0.59 | 0.50 | 0.67 | 0.33 |
| AE-FLOW (Ours) | BTAD | ResNet-Like | 0.57 | 0.44 | 0.53 | 0.97 | 0.09 |
| AE-FLOW (Ours) | BTAD | ConvNet-Like | 0.74 | 0.80 | 0.79 | 0.71 | 0.87 |


Our results show performance similar to that of Zhao et al. on the chest-XRAY dataset when using the ResNet type subnet for the normalizing flow submodule. The utilization of the alternate subnet, which comprises two convolutional layers and a ReLU activation function, resulted in a notable decrease in performance. This was evidenced by the F1-score, which ended up being 13% lower.

The resulting performance of our implemented AE-FLOW model on the OCT2017 dataset is significantly lower than the performance of the original AE-FLOW model by Zhao et al, as shown by the F1-score, which ended up being 21% lower. Additionally, the SPE metric ended up being 8%, compared to the 88% achieved by Zhao et al.

The performance on the BTAD dataset is significantly worse when utilizing the ResNet-like subnet. Using the ConvNet-like subnet increases performance in terms of F1-score by 17%, at the expense of a decrease in SEN of 26%. 


<!--
discuss which experiments we still want to run
which baselines we want to use
-->
## Discussion
Given the experiments conducted, the reproducibility of the results of Zhao et al. seem feasible for some datasets. For large dataset such as OCT2017, performance is found to be significantly worse when compared to the original findings of Zhou et al. 

The performance on BTAD, a dataset outside of the medical domain, seems to be lower than expected. These results are however obtained using the same hyperparameters as Zhao et al. specified. Therefore, an ablation study will have to be conducted in order to investigate whether the performance can be further improved. 

We plan to extend the research by including an uncertainty quantification module. We intend to do so using deep ensemble methods. Furthermore, to gain more insight into the performance of the implented model relative to alternative model, we aim to implement f-AnoGAN (Schlegl et al., 2019). f-AnoGAN is an anomaly detection method, which uses a similar reconstruction based method. f-AnoGAN does not use normalizing flows, giving us valuable insights into the impacts of the normalizing flow module. 


## Contributions
* Jan Athmer - Project implementation, debugging

* Pim Praat - Project implementation, debugging

* Andre de Brandt - Writing notebook/blogpost, debugging

* Farrukh Baratov - Writing notebook/blogpost, debugging

* Thijs Wijnheijmer - Writing notebook/blogpost, debugging


## Bibliography

- Codella, N., Rotemberg, V., Tschandl, P., Celebi, M. E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., Kittler, H., & Halpern, A. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC) (Version 2). arXiv. https://doi.org/10.48550/ARXIV.1902.03368
- Zhao, Y., Ding, Q., & Zhang, X. AE-FLOW: Autoencoders with Normalizing Flows for Medical Images Anomaly Detection. In The Eleventh International Conference on Learning Representations.
- Pang, G., Shen, C., Cao, L., &amp; Hengel, A. V. (2021). Deep Learning for Anomaly Detection. ACM Computing Surveys, 54(2), 1–38. https://doi.org/10.1145/3439950 
- Mishra, P., Verk, R., Fornasier, D., Piciarelli, C., &amp; Foresti, G. L. (2021). VT-ADL: A Vision Transformer network for IMAGE ANOMALY DETECTION and localization. 2021 IEEE 30th International Symposium on Industrial Electronics (ISIE). https://doi.org/10.1109/isie45552.2021.9576231
- Schlegl, T., Seeböck, P., Waldstein, S. M., Langs, G., &amp; Schmidt-Erfurth, U. (2019). F-anogan: Fast unsupervised anomaly detection with generative adversarial networks. Medical Image Analysis, 54, 30–44. https://doi.org/10.1016/j.media.2019.01.010
