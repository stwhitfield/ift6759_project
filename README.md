# Diagnosing Medical Images for COVID-19 Severity

## Abstract

The COVID-19 pandemic has strained healthcare resources and prompted discussion about how machine learning can alleviate physician burdens and contribute to diagnosis. Chest x-rays (CXRs) are used for diagnosis of COVID-19, but few studies predict the severity of a patient’s condition from CXRs. In this study, we produce a large COVID severity dataset by merging three sources and investigate the efficacy of transfer learning using ImageNet- and CXR-pretrained models and vision transformers (ViTs) in both severity regression and classification tasks. A pretrained DenseNet161 model performed the best on the three class severity prediction problem, reaching 80% accuracy overall and 77.3%, 83.9%, and 70% on mild, moderate and severe cases, respectively. The ViT had the best regression results, with a mean absolute error of 0.5676 compared to radiologist-predicted severity scores.

## Results

### Classification task
<img width="422" alt="classification_results" src="https://user-images.githubusercontent.com/14030344/235335107-adc56768-bb85-4985-923e-b79d795a01f6.png">

### Regression task
<img width="422" alt="regression_results" src="https://user-images.githubusercontent.com/14030344/235335149-8ef9874a-9c45-4f2a-a9b2-4165efc05ccc.png">

## Models
- Fine-tuned ViT is available at https://huggingface.co/ludolara/vit-COVID-19-severity.
