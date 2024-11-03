# Retinal-Disease Classification Using Deep Learning EfficientNetB3 and DenseNet201
This repository contains the code and resources for a deep learning project focused on the multiclass classification of retinal diseases using convolutional neural networks (CNNs). The primary objective of this project is to accurately identify four types of retinal conditions: Diabetic Retinopathy, Cataract, Glaucoma, and Normal retina. By leveraging the EfficientNetB3 and DenseNet201 models, this project aims to assist in the early detection and diagnosis of retinal diseases, which is crucial for effective treatment and preventing vision loss.

## Project Highlights
- **Dataset**: A comprehensive dataset of approximately 4000 retinal images, with around 1000 images per class, sourced from public datasets including IDRiD, Ocular Recognition, and HRF.
- **Preprocessing**: Images were resized to 224x224 pixels and normalized for optimal feature extraction. Data augmentation techniques were deliberately excluded to evaluate the baseline performance of the models.
- **Models Used**:
   - **EfficientNetB3**: Known for its efficient scaling, achieving a validation accuracy of 92.89%.
   - **DenseNet201**: Achieved a validation accuracy of 92.42%, providing comparable results with a slightly different architectural approach.
- **Training Details**: Both models were fine-tuned with pre-trained ImageNet weights. The models were trained with the Adamax optimizer, a learning rate of 0.001, for 10 epochs, using an 80:10:10 train-validation-test split.
- **Evaluation Metrics**: Key metrics include accuracy, precision, recall, and F1-score. Diabetic Retinopathy achieved the highest precision score at 99%, highlighting the models’ effectiveness in distinguishing specific conditions.

## Results and Future Directions
The EfficientNetB3 model showed a slight edge in performance due to its compound scaling technique, making it potentially more suitable for clinical applications where both accuracy and computational efficiency are critical. However, the models’ robustness may benefit from additional augmentation techniques and a broader dataset encompassing more retinal disease categories.

## Repository Structure
- **/data**: Contains links or instructions to access the dataset (due to dataset licensing, raw images may not be included).
- **/models**: Pre-trained model weights and model architecture files.
- **/notebooks**: Jupyter notebooks for data preprocessing, model training, and evaluation.
- **/src**: Python scripts for model definitions, training pipelines, and evaluation functions.
- **/results**: Contains saved metrics, plots, and model performance summaries.

## Requirements
-Python 3.8+
-TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib
