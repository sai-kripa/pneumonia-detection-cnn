Pneumonia Detection Using Deep Learning

Project Overview
This project detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN).  
The dataset used is pneumoniamnist.npz, and the model classifies X-ray images as either *Normal* or *Pneumonia*

   Dataset
- File: pneumoniamnist.npz
- Source: MedMNIST
- Image Size: 28x28 (grayscale)
- Classes:
       0 = Normal  
       1 = Pneumonia

Model Description
We used a Convolutional Neural Network (CNN) to detect pneumonia from X-ray images.

The architecture includes:
- Input layer for 28x28 grayscale images  
- Two convolutional layers (with ReLU activation)  
- MaxPooling layers  
- Flatten layer  
- Dense layers for final classification using Softmax

The model is trained on labeled X-ray images to distinguish between normal and pneumonia conditions.

 Tools & Libraries
- Python  
- Google Colab  
- TensorFlow  
- NumPy  
- Matplotlib

Project Workflow

The project follows these steps:

1. Import Libraries
   Load all required libraries such as TensorFlow, NumPy, and Matplotlib.

2.  Load Dataset  
   Use the .npz file to extract training and testing images and labels.

3.   Preprocess Data 
   - Normalize image values  
   - Reshape data for CNN input

4. Build CNN Model
   Define the layers using Keras Sequential API.

5. Compile Model 
   Use:
   - Optimizer: Adam  
   - Loss Function: Categorical Crossentropy  
   - Metrics: Accuracy

6. Train Model
   Fit the model on training data over multiple epochs with validation.

7. Evaluate Model  
   Evaluate on test data to calculate accuracy:
  
   test_loss, test_acc = model.evaluate(X_test, y_test)
   print("Test Accuracy:", test_acc)

8. Save Model 
   save the trained model for future use