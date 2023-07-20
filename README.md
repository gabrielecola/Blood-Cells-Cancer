## Blood Cells Cancer analyzed via CNN

<div align="center">
  <a href="https://huggingface.co/spaces/statgabriele/bloodcancer">
    <img src="https://huggingface.co/favicon.ico" alt="Hugging Face" width="70" height="70">
    <h3 style="font-size: 10px;">Explore the Web App on Hugging Face</h3>
  </a>
</div>



## Table of Contents

 - [Blood Cells Cancer analyzed via CNN](#blood-cells-cancer-analyzed-via-cnn)
- [1. Problem Statement](#1-problem-statement)
- [2. Data Description](#2-data-description)
  * [Attribute Information](#attribute-information)
    + [Inputs](#inputs)
    + [Output](#output)
- [3. EDA](#3-eda)
- [4. Modelling Evaluation](#4-modelling-evaluation)
- [5. Results](#5-results)

### Blood Cells Cancer analyzed via CNN

The definitive diagnosis of Acute Lymphoblastic Leukemia (ALL), as a highly prevalent cancer, requires invasive, expensive, and time-consuming diagnostic tests. ALL diagnosis using peripheral blood smear (PBS) images plays a vital role in the initial screening of cancer from non-cancer cases. 
Thus , our aim is to use two different deep learning architectures, CNN,  in order to classiify correctly all stages of cancer.



### 1. Problem Statement
Classify correctly the 4 classes of stages of cancer, where one of them is benign and the others three are Malignant

### 2. Data Description
Data is obtained from  [kaggle](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class).

- Number of instances - 3242
- Number of classes - 4
  #### Attribute Information
  ##### Inputs
  - filepath: filepath of images
  ##### Output
  - label :  classification of 4 types of cells cancer
  

   
  
 ### 3. EDA
 <p float="left">
  <img src="https://user-images.githubusercontent.com/103529789/210671064-c1f3278f-1fc5-4d55-9f35-c4d2df4a1fc5.png" width="350"/>
  <img src="https://user-images.githubusercontent.com/103529789/210671106-75de3e10-b60d-4b9d-b1e8-0f54ddd5d38d.png" height='315' width="450"/>
  </p>
  



  
  
 ### 4. Modelling Evaluation
 - Algorithms used
    - AlexNet
    - VGG16
 - Metrics used: Accuracy, Precision, Recall, F1-Score
 
  ### 5. Results
  
   <p float="left">
  <img src="https://user-images.githubusercontent.com/103529789/210671268-710cc54b-27bc-4ce1-a0db-2615484b1e47.png" width="380"/>
  <img src="https://user-images.githubusercontent.com/103529789/210671318-f4c36f5e-28d1-48f5-8418-42524e1189c5.png" width="380"/>
  </p>
  

