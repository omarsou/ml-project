# Tuberculosis Classification

This project is related to the zindi Africa Hackathon "Tuberculosis Classification via X-Rays Challenge" (You can look at the challenge here : https://zindi.africa/competitions/runmila-ai-institute-minohealth-ai-labs-tuberculosis-classification-via-x-rays-challenge)

## Summary
In this project, we have to build an AI model that can classify Tuberculosis and Normal X-Ray results. The training set contains 353 images of x-rays of TB-positive lungs and 365 images of x-rays of healthy lungs. The test set contains 82 images.
There are two datasets, the small one (reduced resolution) and the big one.
I choose the small one because the other one is too large.

## Folder Description

### tuberculose_notebook.ipynb
In this notebook, I describe my approach based on transfer learning.

## Results

### sub_base.csv
Used VGG16 + top basic deep learning architecture : Score 0.927601

### sub_0.csv
Used VGG16 (all the layer no trainable) + attention based architecture : Score 0.9638001

### sub_1.csv
Used VGG16 (last block conv layer trainable) + attention based architecture : Score 0.927601

### sub_high_res.csv
Same thing as sub_0 except that for this one I resized my image to (500,500) (size image for the previous one : (254,254))

### sub_high_attention.csv
Implementation (Convolutional Block Attention Module) from https://github.com/kobiso/CBAM-keras, Score : 0.773755

@TODO: 
- Try grayscale image
- Try ensemble model (according to this article : https://pubs.rsna.org/doi/10.1148/radiol.2017162326 : 
" The best-performing classifier had an AUC of 0.99, which was an ensemble of the AlexNet and GoogLeNet DCNNs." )
-Try to connect CBAM & VGG16