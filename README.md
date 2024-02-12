
## Transfer learning for bone lesion classification 

I fine-tuned four different DCNN models to classify bone lesions caused by multiple myeloma.The best model, ResNet50, achieved an F1-score of 0.83 on the test set. 

![image](https://github.com/munirdin87/bone_lesion_classification/assets/49895184/fb704135-bdcb-454d-8098-4401d0a0c4b0)

## Link to the paper: 
https://bnaic2023.tudelft.nl/static/media/BNAICBENELEARN_2023_paper_65.7e5de9cf01a9bf3f4bc8.pdf

# Transfer leaning model architecture 
Transfer learning involves replacing the final classification layers of a pre-trained model with custom ones. A Global Average Pooling (GAP) layer is preferred over a flatten layer to reduce overfitting, especially with limited data. After this modification, additional dense and output layers, often with sigmoid activation, are added to the model.

![image](https://github.com/munirdin87/bone_lesion_classification/assets/49895184/7b5f6c0f-9c77-4104-8d1e-1ab19d99294e)

## Findings 
* Objective: Reduce false positives in osteolytic lesion segmentation.
* Context: MM patients with osteolytic lesions.
* Approach: Automated segmentation followed by deep learning classifiers.
* Model Performance: Fine-tuned ResNet50 achieved F1 score of 0.83.
* Clinical Impact: Enhances reliability in lesion detection.
