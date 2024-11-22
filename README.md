# Face Recognition Model Training and Analysis

This project explores the training and evaluation of face recognition models using dimensionality reduction techniques such as **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)**. It uses the **Labeled Faces in the Wild (LFW)** dataset as a benchmark for performance analysis.

---

## Table of Contents
1. [Objectives](#objectives)
2. [Techniques Used](#techniques-used)
    - [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
    - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
3. [Dataset](#dataset)
4. [Preprocessing](#preprocessing)
5. [Implementation](#implementation)
6. [Results](#results)
7. [Insights and Recommendations](#insights-and-recommendations)
8. [References](#references)
9. [Code Repository](#code-repository)

---

## Objectives

The primary goal of this project is to train a face recognition model that can:
- **Accurately identify or verify individuals** based on facial features.
- Perform **dimensionality reduction** to reduce computational complexity.
- Extract **discriminative features** for better class separability.
- Ensure **real-time performance** for practical applications.

---

## Techniques Used

### Linear Discriminant Analysis (LDA)
- **Purpose:** Maximize class separability by projecting data onto a lower-dimensional space.
- **Advantages:**
  - Enhances class separation for better classification performance.
  - Requires fewer components than PCA, reducing computational costs.
- **Limitations:**
  - Requires class labels and assumes linear separability.
  - Less effective with very high-dimensional datasets without preprocessing.

### Principal Component Analysis (PCA)
- **Purpose:** Capture the maximum variance in data by reducing it to principal components.
- **Advantages:**
  - Effective for noise reduction and dimensionality reduction.
  - Unsupervised, requiring no class labels.
- **Limitations:**
  - Does not focus on class separation, which may reduce classification accuracy.

---

## Dataset

The **Labeled Faces in the Wild (LFW)** dataset is used, containing over 13,000 labeled images of faces from the web.  
- **Challenges:**
  - High variability in lighting, pose, and expressions.
  - Class imbalance, with some individuals having significantly more images than others.
- **Significance:** A widely used benchmark for face recognition research.

---

## Preprocessing

1. **Resizing:** Standardized image dimensions for consistency (resize = 0.4).  
2. **Grayscale Conversion:** All images are already grayscale, simplifying preprocessing.  
3. **Data Standardization:** Standardized pixel intensities to have zero mean and unit variance.  
4. **Dimensionality Reduction:**  
   - PCA: Reduced features to capture maximum variance.  
   - LDA: Reduced features to enhance class separability.

---

## Implementation

### Tools and Libraries
- **Python Libraries:** `scikit-learn`, `matplotlib`, `numpy`, `scipy`
- **Classifiers:** Support Vector Machine (SVM) with RBF kernel.

### Steps
1. **Data Loading:**  
   - Fetch the LFW dataset using `fetch_lfw_people`.  
2. **Train-Test Split:**  
   - Data split into 70% training and 30% testing.  
3. **Dimensionality Reduction:**  
   - PCA and LDA applied to training data.  
4. **Classification:**  
   - SVM classifier trained on reduced dimensions.  
   - Hyperparameter tuning using `RandomizedSearchCV`.

---

## Results

### Baseline Model (No Dimensionality Reduction)
- **Accuracy:** 85%
- **F1-Score:** 85%

### PCA-Based Model
- **Accuracy:** 82%
- **F1-Score:** 82%

### LDA-Based Model
- **Accuracy:** 69%
- **F1-Score:** 70%

---

## Insights and Recommendations

### Observations
1. **Dimensionality Reduction Impact:**
   - **PCA** captures overall variance but may miss class-specific separability.
   - **LDA** excels at class discrimination but may lose variance crucial for complex datasets like LFW.
2. **Computational Efficiency:**
   - PCA becomes computationally expensive with high component counts.
   - LDA is generally more efficient for fewer components.

### Recommendations
- **Hybrid Approach:** Apply PCA to reduce dimensionality, followed by LDA for class separation.  
- **Task-Specific Application:**  
   - Use PCA for tasks focusing on general data structure or noise reduction.  
   - Use LDA for classification tasks with well-separated classes.  
- **Big Data Optimization:** Consider incremental PCA or online LDA for scalability.  

---

## References

1. Wikipedia contributors. (2024). *Linear discriminant analysis*. Retrieved from [Wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
2. Scikit-learn documentation. (2024). *LinearDiscriminantAnalysis*. Retrieved from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
3. Martinez, A. (2011). *Fisherfaces*. Scholarpedia, 6(2), p. 4282. doi: [10.4249/scholarpedia.4282](http://www.scholarpedia.org/article/Fisherfaces)
4. Seymur. (2024). *Face Recognition (Eigenfaces, Fisherfaces, LBPH)*. Medium. Retrieved from [Medium](https://medium.com/@seymurqribov05/face-recognition-eigenfaces-fisherfaces-lbph-0b39d41bd54c)
5. Data Headhunters (2024). *PCA vs LDA: Dimensionality Reduction Techniques Explored*. Retrieved from [Data Headhunters](https://dataheadhunters.com/academy/pca-vs-lda-dimensionality-reduction-techniques-explored/)
