---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Montserrat', sans-serif;
    color: #333333;
  }
  h1, h2 {
    color: #A01246;
    font-weight: 700;
  }
  h3 {
    color: #5A4A82;
  }
  strong {
    color: #C40067;
  }
  hr {
    border: 0;
    border-top: 3px solid #C40067;
    width: 60%;
    margin: 1.2em auto;
  }
---
<!-- SLIDE 1 -->
# **Explainable AI and Facial Rcognition**
## do we have a subtitle

**Group 5**
**Names:** Laia Domenech Burin, Sofiya Berdiyeva, Giulia Maria Petrilli, Fanus Ghorjani
**Date:**  12th December 2025

---
<!-- SLIDE - What is Explainable AI? -->

# What is Explainable AI?

- Makes ML models and predictions **understandable** for humans  
- Needed in **high-stakes public policy** domains  
  (e.g., security, justice, healthcare, social welfare)
- Users are not only data scientists → also **authorities & decision-makers**
- Explanations must **support real decisions** and human accountability

> Source: Amarasinghe et al., 2020
---
<!-- SLIDE - Why Explainable AI matters -->
# Why Explainable AI matters

- In sensitive contexts, AI bias can cause **unequal treatment & human-rights violations**, especially for racialized groups (George, 2022)  
- Errors are not neutral → misclassifications decide **who passes, who gets stopped, who gets flagged** — risk of **«discrimination by default»**  
- In border & migration contexts, such systems work at the **“border of rights”** — racism and xenophobia risks are particularly high  
- States have an obligation to protect people from discrimination by ensuring **regulation, oversight & transparency** of AI systems  
---
<!-- SLIDE - Policy Relevance -->

# Relevance for Policy Context

- Facial Recognition is increasingly used in **public security & border control**
- Example Germany: test deployments show risks of **surveillance, profiling and discriminatory misclassification** (Töper & Kleemann, 2025)
- Globally applied in migration & security contexts **without sufficient regulation** or fairness checks (Lynch, 2024)

> FRT can directly affect rights and freedom → bias becomes a **policy and human-rights issue**.


---
<!-- SLIDE - What is Fairness? -->

# What is Fairness?

- A model should **not systematically disadvantage** any person or demographic group
- Two main perspectives:
  - **Group Fairness** → equal treatment across protected groups
  - **Individual Fairness** → similar individuals receive similar outcomes
- Example: higher misclassification rates for Black persons in risk categorization  
  (Özmen Garibay & Gallegos, 2022)

> Fairness ensures **equity** in automated decision-making.
---
<!-- SLIDE 5 - Descriptives 1 -->
<style>
section {
  font-size: 2em;
  line-height: 1.25;
}
</style>

# Descriptives — Dataset Summary

- **8 demographic sub-groups**:
  - asian_females, asian_males  
  - black_females, black_males  
  - indian_females, indian_males  
  - white_females, white_males

- **2500** samples per group → **20,000 images** total
- Avg. dimensions: ~108×124 px
- Protected attribute: **Ethnicity × Gender**
---
<!-- Slide : Descriptives 2 -->
<!-- SLIDE - Descriptives 2 -->

# Descriptives — Visual Overview
<style>
img {
  width: 85%;
  display: block;
  margin: 0 auto;
}
</style>

- Representative **mean faces** per demographic subgroup  
  (Ethnicity × Gender)

<img src="figures/mean_faces_all_withlabels.png">

> Noticeable visual variation across sub-groups


---
<!-- SLIDE - Metrics to Assess Bias (I) -->
# Metrics to Assess Bias (I)
<style>
img {
  width: 70%;
}
</style>

- Precision
- Accuracy
- F1: represents model's overall ability to recognize a class bias.

![Per-class performance metrics](figures/per_class_performance_metrics.png)

---
<!-- SLIDE - Metrics to Assess Bias (II) -->
# Metrics to Assess Bias (II)

- Fairness across demographic groups assessed using:
  - **Demographic Parity (DP)**
  - **Equal Opportunity (TPR)**
  - **Individual Fairness Proxy**

<div style="text-align:center;">
  <img src="figures/fairness_evaluation.png" width="95%">
</div>

> Takeaway: Model shows **clear group fairness disparities**.
---


<!-- SLIDE - Key Bias Findings -->
# Key Bias Findings

- **Big gap** in accuracy (~15pp)
- **Best:** Black males  
- **Worst:** Asian males / White males / White females  

- **DP violation:** Asian males → strong over-prediction  
- **TPR inequity:** White & Asian males disadvantaged  
- **Individual inconsistency:** Black & white females highest variation  

> Conclusion: Model is **not equitable** across demographic intersections.

---

<!-- SLIDE 8 -->
# Libraries
<hr>

- [Placeholder for tools]
- [Placeholder for references]

---

<!-- SLIDE 9 -->
# Xplique
<hr>

- miau
- miau

---
<!-- SLIDE - Explain NNs -->

# Explain NNs

- Open-source library for **Explainable AI** in neural networks  
- Focus on **latent space analysis** and **uncertainty estimation**  
- Works for **classification networks** like our model  

---

### Why we use Explain NNs

- Our results show **performance differences between groups**
  → we need to understand **why** the model performs worse for some groups  
- XAI helps to **open the black box** and detect hidden bias

- We applied:
  - **Saliency Maps** → shows activated facial regions
  - **Integrated Gradients (IG)** → more **reliable** feature attribution over multiple pixel paths

> Goal: see **how** the model decides → reveal potential bias in the decision process
--- 
# Bias assessment 

map


--- 
# Bias assessment 

- Inconsistent feature attribution for Asian males and White females
- Limited reliance on facial periphery, especially for female examples
- Uneven or asymmetric activation for White males
- Group-specific deviations from the general IG pattern
---


<!-- SLIDE 11 (Optional) -->
# Thank You!
Questions?

--- 

# Sources

---