# Diagnosing Medical Images for COVID-19 Severity

## Abstract

The COVID-19 pandemic has strained healthcare resources and prompted discussion about how machine learning can alleviate physician burdens and contribute to diagnosis. Chest x-rays (CXRs) are used for diagnosis of COVID-19, but few studies predict the severity of a patient’s condition from CXRs. In this study, we produce a large COVID severity dataset by merging three sources and investigate the efficacy of transfer learning using ImageNet- and CXR-pretrained models and vision transformers (ViTs) in both severity regression and classification tasks. A pretrained DenseNet161 model performed the best on the three class severity prediction problem, reaching 80% accuracy overall and 77.3%, 83.9%, and 70% on mild, moderate and severe cases, respectively. The ViT had the best regression results, with a mean absolute error of 0.5676 compared to radiologist-predicted severity scores.

## Datasets

We combined three datasets for which there were severity scores available. The ieee8023 Covid-19 Image Data Collectiont (Cohen et al., 2020c), which we will refer to as the Cohen dataset, currently contains 481 images from covid-positive patients in PA (posterior-anterior) and AP (anterior-posterior) modalities, 93 of which have associated severity scores in the form of opacity scores. This dataset was manually curated and collected from sources including journal publications, and contains samples from across the globe. It has been expressly designed to be suitable for ML tasks and contains not only X-ray images but also physician annotations, collection data and metadata useful for prediction tasks. The Brixia score dataset (Signoroni et al., 2021), contains 4707 images of COVID-positive patients from Northern Italy with Brixia severity scores, in AP and PA projection. The data contain all the variability of a real clinical scenario, since they consist of all CXR images taken in sub-intensive and intensive care units during a month-long period of pandemic peak in the ASST Spedali Civili di Brescia. The RALO Stony Brook Medicine dataset (Cohen et al., 2021a) from the northeastern United States contains 2373 covid-positive images and associated opacity scores. All of these datasets have been scored by expert radiologists for severity assessment. Figure 3 shows the distributions of severity scores for each dataset in stacked histogram form. Although the dataset is imbalanced, it is representative of a real-world scenario.

<p align="center">
  <img width="375" alt="Screenshot 2023-04-30 at 0 22 39" src="https://user-images.githubusercontent.com/14030344/235335405-4a1a2939-c894-4288-b9fe-6a2bb27ec8f4.png">
</p>

## Models

Pretrained models used in this study are given in Table 3. Image processing systems usually use convolutional neural networks (CNNs) which, due to the nature of the convolution operation, permit the model to invariantly learn spatial patterns while keeping the number of parameters relatively low. AlexNet (Krizhevsky et al., 2012) was a groundbreaking model in computer vision, showing that convolutions were a powerful tool for image recognition. SqueezeNet (Iandola et al., 2016) achieved similar results to AlexNet while having 50x fewer parameters due to architectural choices. MobileNetv2 (Sandler et al., 2019) is a lightweight CNN with inverted residual structure modules. VGG-16 (Simonyan & Zisserman, 2015) is a high-performing CNN with a deep architecture, using 3x3 kernel filters but 16 convolutional layers for a relatively simple but expressive network. DenseNet (Huang et al., 2018) models have layers that spread their weights over multiple inputs. This means that deeper layers can use features extracted early on, cutting down the total number of parameters and allowing very deep models. DenseNet-121, used extensively in this study, contains 120 convolutional layers and a final fully-connected layer.

VGG-16, AlexNet, DenseNet, MobileNet v2 and SqueezeNet models were imported from pytorch with pretrained weights. DenseNet121 variants were imported from the torchxrayvision library (Cohen et al., 2021b) with pretrained weights (all, chex, pc, mimic nb, mimic ch, nih, and rsna). Each of these models are pretrained on a different source of CXRs, with datasets of varying sizes (see Table 3 for sizes). All model weights other than those in classification layers were frozen. The final output layers of ImageNet-pretrained models were adjusted to output 1 value for the regression task, or 3 values for the classification task. For torchxrayvision models we added an extra two linear layers with a ReLU activation function in between to adapt the 18-class output of the model to our regression or classification tasks.

A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. Transformers found their initial applications in natural language processing (NLP) tasks (Devlin et al., 2019; Brown et al., 2020). The Vision Transformer (ViT) (Alexey et al., 2020) computes relationships among pixels in various small sections of the image (e.g., 16x16 pixels), at a drastically reduced cost. Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. The result is fed to the transformer and attention mechanisms applied. We imported a ViT from Hugging Face pretrained on 14 million images and fine-tuned it for 20 epochs on our dataset.

<p align="center">
  <img width="443" alt="Screenshot 2023-04-30 at 0 38 23" src="https://user-images.githubusercontent.com/14030344/235335777-7b885236-ea8d-432a-9d93-ae18388afd07.png">
</p>

## Results and Analysis

Results for the severity classification task are shown in Table 4. While we began experiments with seven distinct classes (corresponding to the severity score), we reached only 61% classification accuracy with seven classes. We believe that was largely due to the imbalance of the dataset. When we approached binned the severity labels into three categories, ”mild”, ”moderate” and ”severe” , our models performed better (Table 4). Our model performed best on the dominant class (”moderate”). Despite the VGG-16 having more trainable parameters, the DenseNet161 had better accuracy, highlighting the idea that architectural decisions are important.

The DenseNet161 model was able to distinguish mild cases quite well, but we saw some confusion between moderate and severe cases (Figure 3). We attribute these mistakes to the imbalanced nature of the dataset: it is possible these true values fall on the extremity of the bins. This is a consideration for future work. Results for the severity regression task are given in Table 6. We obtained slightly better MAE, MSE and R2 values than Signoroni et al. 2021, which we attribute to using a larger dataset. The agreement between our models’ predictions and the radiologist predictions, indicated by the R2 value, only reached around 50%. Since other studies have reported values similar to ours (Cohen et al., 2020a; Signoroni et al., 2021), it seems likely that this is a very hard problem and that there are other factors that a purely image-based prediction model cannot take into account.

Comparing the fine-tuning of models that are pretrained on CXRs and on general images (ImageNet), it is clear that pretraining on CXRs provides a discriminative boost to the model. Perhaps surprisingly, the SqueezeNet model performed as well or better than some CXR-pretrained models. These models were all trained on fewer than 50,000 images, suggesting that there is an approximate minimum of CXRs that are needed for the domain-specific prediction boost we see here. Still, pretraining on CXRs clearly improved results more than having an architecture with a high capacity: VGG-16 and AlexNet models did poorly in the regression task despite having a high number of trainable parameters.

The ViT had the best predictions in the regression task, with an MAE of 0.5676. We believe that this number is adequate in giving physicians a good indication of the severity of the COVID-pneumonia experienced by their patients. It is unclear to us whether the impressive results of the ViT compared to other models are due to the sheer number of images upon which the ViT was trained (14 million) or whether attention mechanisms provide a particular boost. Using a ViT pretrained from scratch or on a smaller dataset might help to answer this question. We also cannot rule out the possibility that a good subset of the 14 million images in the ViT’s dataset are medical images, which would also help the model perform better without this improvement being due to attention-based and architectural factors.

### Classification task

<p align="center">
  <img width="422" alt="classification_results" src="https://user-images.githubusercontent.com/14030344/235335107-adc56768-bb85-4985-923e-b79d795a01f6.png">
</p>

### Regression task
<p align="center">
  <img width="422" alt="regression_results" src="https://user-images.githubusercontent.com/14030344/235335149-8ef9874a-9c45-4f2a-a9b2-4165efc05ccc.png">
</p>

### Explainability

Being able to explain an ML model’s decisions is a key part of ensuring stakeholder confidence in the model. We generated saliency maps (Figure 4) using our DenseNetall torchxrayvision pretrained model to help us understand which regions of the image are influencing the model’s decisions. Different images had different focal points, suggesting that the model is looking at features of the images and not purely shortcut learning. Without consulting experts (radiologists), however, it is difficult to judge whether the model is focusing on the right regions.

<p align="center">
  <img width="421" alt="Screenshot 2023-04-30 at 0 28 42" src="https://user-images.githubusercontent.com/14030344/235335521-46111685-0bfb-43c7-957d-aab5f46cfa51.png">
</p>

### Models
- Fine-tuned ViT is available at https://huggingface.co/ludolara/vit-COVID-19-severity.

## Conclusion
In conclusion, we have leveraged the power of machine learning methods for predicting the severity of the COVID19 condition of a patient from the chest X-rays (CXRS) by generating a large single dataset from three different data sources. We have approached the problem as regression and classification severity prediction tasks and used different pre-trained ImageNet-models VGG-16, AlexNet, DenseNet, MobileNet V2, SqueezeNet, vision transformer (ViT) and chest-X ray pre-trained model DenseNet-121.

DenseNet161 and VGN16 pretrained models with fine tuning achieved an accuracy of 80% and 78% with three classes (low, medium, high) and 62% and 61% with seven classes (0-6 scale). We have found that all models exhibit worst performances without fine tuning. Moreover, we did not obtain remarkable results for seven classes for ViT. Further dissection of ”edge cases” is necessary to understand how many categories of severity our models should be able to reliably predict. For regression tasks, we found X-ray pretrained models generally exhibited superior performance over ImageNet pre-trained models. 

Our results are competitive with others in the COVID severity prediction space (Zandehshahvar et al., 2021; Tabik et al., 2020; Signoroni et al., 2021; Cohen et al., 2020a) without training from scratch or using more complicated architectures. Further work could build on our framework and determine exactly how much of an improvement is added by lung segmentation, as has been done in other studies (Danilov et al., 2022; Tabik et al., 2020; Wang et al., 2021b). The ViT has the best result for regression with the lowest MSE value of 0.5135. Further work could determine what properties of the ViT made it more successful than other models for our task. We also developed saliency maps using the DenseNet-all torchxrayvision pre-trained model which help us to identify which regions of the image are influencing the model’s decisions, although other approaches such as GRAD-CAM (Selvaraju et al., 2016) could be explored as complementary ways of improving stakeholder confidence in machine learning predictions in radiology.

## Contributions of Each Team Member
- Lucia Eve Berger performed the VGG-16 baseline (binary classification) experiment, severity classification with VGG16 and densenet CXR models, and contributed the classification part of the following written sections: Baselines, Evaluation Methods, Experimental Results, and Results and Analysis.
- Luis Lara performed dataset preprocessing, creation of testing modules, investigation of efficacy of MLPMixer model and ViT for a seven-class prediction task, produced the results for the ViT model for the regression task, compiled Table 3, contributed the ViT part of the Experimental Results section and contributed ideas to the Results and Analysis regression discussion.
- Rajesh Raju investigated data preprocessing including image and text augmentation methods, investigated the utility of clinical notes, resized CXR images to 224 x 224, and wrote part of the conclusion.
- Shawn Whitfield combined the three CXR datasets, including extracting .dcm files and mapping severity scores into opacity scores, built the PyTorch Lightning framework to facilitate train:validation cycles, performed all regression experiments on pretrained ImageNet and torchxrayvision models (other than the ViT), implemented the saliency mapping, and wrote the following sections of the paper: Abstract; Introduction; Related work; Methods - data augmentation, models, saliency maps; Experiments - datasets, evaluation methods, experimental results (regression), results and analysis (regression, saliency); part of the conclusion.
