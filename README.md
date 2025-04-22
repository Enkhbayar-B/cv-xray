## 🧠 UNet-Based Image Classifier for X-ray Diagnosis

This project implements an image classification system using a UNet-inspired convolutional neural network in PyTorch. </br>
It supports:
 - Custom training with penalized loss (False Positive/Negative aware)</br>
 - Easy model saving and loading</br>
 - Folder-based image classification inference</br>
 ![Screenshot 2025-04-16 194227](https://github.com/user-attachments/assets/9016fb85-bc6a-49cd-ab59-c8fcdef45f1f)


## 📁 Project Structure

unet/</br>
├── train_unet.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Train and validate the UNet model</br>
├── unet_model.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# UNetClassifier model + custom loss</br>
├── infer_unet.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Inference script for predicting new images</br>
├── unet_classifier.pth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Saved trained model (generated after training)</br>
├── README.md&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Project documentation</br>
└── xray/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Image dataset folder (same level as scripts)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train/</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── val/</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── test/</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── inf/</br>
</br>
## 🧠 1. unet_model.py – Model & Custom Loss

This module defines:</br>
UNetClassifier: A downsampling-only UNet encoder model adapted for image classification.</br>
FPFNPenalizedCELoss: Custom loss that applies higher penalties to false positives and false negatives.</br>
</br>
## 🏋️ 2. train_unet.py – Train the Model

Trains the model using the data in xray/train and xray/val, validates accuracy and confusion matrix every few epochs, and saves the model to unet_classifier.pth.

<pre> Run the training script with `python train_unet.py`. </pre></br>

This will:</br>
 - Train for 100 epochs</br>
 - Print validation loss and accuracy</br>
 - Show confusion matrix every 10 epochs</br>
 - Save the model to unet_classifier.pth</br>
 
 </br>

 ## 📊 Training Configuration Summary

<pre>
╒═══════════════════════╤═══════════════════════════════════════════════════════════════════╕
│ Parameter             │ Value                                                             │
╞═══════════════════════╪═══════════════════════════════════════════════════════════════════╡
│ Model Architecture    │ UNetClassifier                                                    │
│ Input Channels        │ 3                                                                 │
│ Output Classes        │ 2                                                                 │
│ Feature Sizes         │ [256, 512, 1024, 2048, 4096]                                      │
│ Loss Function         │ FPFNPenalizedCELoss                                               │
│ False Negative Weight │ 3.0                                                               │
│ False Positive Weight │ 3.0                                                               │
│ Optimizer             │ Adam                                                              │
│ Learning Rate         │ 0.001                                                             │
│ Weight Decay          │ 1e-05                                                             │
│ Batch Size            │ 4                                                                 │
│ Epochs                │ 100                                                               │
│ Train Augmentations   │ Resize, RandomHorizontalFlip, RandomRotation, ToTensor, Normalize │
│ Val/Test Transforms   │ Resize, ToTensor, Normalize                                       │
│ Normalization Mean    │ [0.0, 0.0, 0.0]                                                   │
│ Normalization Std     │ [1.0, 1.0, 1.0]                                                   │
│ Train Dataset Path    │ /home/bay/codes/unet/xray/train                                   │
│ Val Dataset Path      │ /home/bay/codes/unet/xray/val                                     │
│ Test Dataset Path     │ /home/bay/codes/unet/xray/test                                    │
│ Device                │ cuda                                                              │
│ Model Save As         │ unet_classifier.pth                                               │
╘═══════════════════════╧═══════════════════════════════════════════════════════════════════╛
</pre>
 
 ## 🔍 3. infer_unet.py – Run Inference
 
Run predictions on all images in the xray/inf/ directory using the saved model.

<pre> Run the training script with `python infer_unet.py`. </pre></br>
## ✅ Output
Console:
<pre>
img001.jpg --> pneumonia
img002.jpg --> normal</pre>

File: xray/inf/predictions.txt
<pre>
img001.jpg	pneumonia
img002.jpg	normal</pre>

## 📝 Notes

 - The model expects input images to be RGB and sized to 256×256.
 
 - Make sure the label classes (normal, pneumonia) in train/val/test are consistent and match the order used in training.
 
 - Inference will skip over non-image files and log any processing errors without crashing.
 
## 🔧 Customization

You can easily tweak:

 - features in the UNet model (more/less depth)

 - FN/FP weights in the loss function

 - Number of output classes

 - Data augmentations or normalization in train_unet.py
