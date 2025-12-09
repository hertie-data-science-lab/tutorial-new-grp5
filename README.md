# Explainable-AI-XAI-and-facial-recognition
This tutorial focuses on explainable AI (XAI) and bias of the models. We believe that explainability of a model can provide insights into why it can be constructing biased decisions, and how to prevent them. Within this approach, we replicate a resnet18 model from the paper [\'Face Recognition: Too Bias, or Not Too Bias?\'] (https://doi.org/10.48550/arXiv.2002.06483) (Robinson et al., 2020) and demonstrate the usage of two packages that provide tools for explainable AI purposes: Xplique and Captum.

Tutorial goes through various bias assessment techniques, gives a quick overview of attribution methods and metrics for those methods' evaluation.

To download the data with classified face images, please use this [Dropbox link](https://www.dropbox.com/scl/fi/5gindh41lrw8j7bgyv9mq/BFW-Release.zip?rlkey=k7kf4knhm18qi3be661m8qmo4&e=2&st=w5k6o36d&dl=0%3E)

Since Xplique is mainly working with Tensorflow, which is not supported by the latest versions of Python, please use Python 3.9–3.12.

# References
Robinson, J. P., Livitz, G., Henon, Y., Qin, C., Fu, Y., & Timoner, S. (2020). Face Recognition: Too Bias, or Not Too Bias? (No. arXiv:2002.06483). arXiv. https://doi.org/10.48550/arXiv.2002.06483

Tutorials—Xplique. (n.d.). Retrieved December 9, 2025, from https://deel-ai.github.io/xplique/latest/tutorials/

