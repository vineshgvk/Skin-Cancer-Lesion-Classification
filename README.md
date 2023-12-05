# Skin Cancer Image Classification

## Overview

This project uses a convolutional neural network (CNN) model to classify skin images into 9 classes of skin cancers and lesions. The model is trained on the Skin Cancer dataset which contains over 2,000 images across the following classes:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis  
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesions

The images are split into training and validation sets, with image augmentation performed on the training set to handle class imbalance. A CNN model with convolution, pooling, dropout, and dense layers is defined and trained for 50 epochs. Key metrics like accuracy and loss are tracked during training and evaluated on the test set after training.

## Usage

The main Jupyter notebook is `skin_cancer_classification.ipynb`. To use the trained model to make predictions, modify the prediction code at the end of the notebook:

```python
test_image_path = #path to test image
test_image = load_img(test_image_path, target_size=(180, 180, 3)) 

# Make prediction
img = np.expand_dims(test_image, axis=0)
pred = model.predict(img)  
pred_class = class_names[np.argmax(pred)]
```

The model achieves over 85% validation accuracy. Performance can likely be improved with additional data augmentation and hyperparameter tuning.

## Results

The model achieves the following results:

- Training Accuracy: 94.58%
- Validation Accuracy: 85.3% 
- Validation Loss: 0.7674

## Requirements

The main Python packages used are:

- Tensorflow 2.0+
- Matplotlib
- Numpy 
- Augmentor (for image augmentation)
