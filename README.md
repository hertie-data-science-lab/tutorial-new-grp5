# Explainable AI and Facial Recognition

This tutorial focuses on explainable AI (XAI) and bias of the image classification models. We believe that explainability of a model can provide insights into why it can be constructing biased decisions, and how to prevent them. Within this approach, we replicate a [resnet18 model](https://github.com/visionjo/facerec-bias-bfw?tab=readme-ov-file) from the paper '[Face Recognition: Too Bias, or Not Too Bias?](https://doi.org/10.48550/arXiv.2002.06483)' (Robinson et al., 2020) and demonstrate the usage of two packages that provide tools for explainable AI purposes: Xplique and Captum.

In the tutorial, we:

1. Go through various bias assessment techniques and metrics

2. Give a quick overview of attribution methods and metrics for those methods' evaluation

3. Apply a naive approach to qualitatively analyze a small sample of images through attribution/saliency maps to derive differences between highlighed regions among classes.

To download the data with classified face images, please use this [Dropbox link](https://www.dropbox.com/scl/fi/5gindh41lrw8j7bgyv9mq/BFW-Release.zip?rlkey=k7kf4knhm18qi3be661m8qmo4&e=2&st=w5k6o36d&dl=0%3E)

Since Xplique is mainly working with Tensorflow, which is not supported by the latest versions of Python, please use Python 3.9–3.12.

# Contributions:

**Giulia Maria Petrilli** - wrote helpers code for data and model loading, preprocessing and bias assessment, prepared an in-class presentation together with Fanus, reviewed and tested code and added text throughout tutorial preparation. You can track related GitHub commits in the current AND [our first GitHub repo](https://github.com/GiuliaGGG/Explainable-AI-XAI-and-facial-recognition/commits/main/), that was used before the official link was fixed.

**Laia Domenech Burin** - refactored and modularized EDA code, created the section of the same name in the notebook with plots and accompanying text, debugged the final code and enriched interpretations. Last but not least, starred in the main tutorial video. 

**Fanus Ghorjani** - created a code and text about Captum library, worked on an in-class presentation with Giulia.

**Sofiya Berdiyeva** - prepared parts related to Xplique library and naive attribution maps analysis, README.md and requirements.txt.


# References

Robinson, J. P., Livitz, G., Henon, Y., Qin, C., Fu, Y., & Timoner, S. (2020). Face Recognition: Too Bias, or Not Too Bias? (No. arXiv:2002.06483). arXiv. https://doi.org/10.48550/arXiv.2002.06483

Tutorials—Xplique. (n.d.). Retrieved December 9, 2025, from https://deel-ai.github.io/xplique/latest/tutorials/

