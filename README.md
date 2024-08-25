EEG-Based Early Detection of Neurological Disorders
Abstract
This repository contains research and code for the early detection of neurological disorders such as Alzheimer's and Parkinson's disease using EEG data and advanced machine learning techniques. Our models leverage the InceptionV3 architecture combined with LSTM layers for Alzheimer's detection, achieving remarkable accuracy. For Parkinson's, we evaluated multiple machine learning classifiers, identifying K-Nearest Neighbors (KNN) as the most effective model.

Project Features
Alzheimer's Detection: Using EEG data and deep learning models, specifically the InceptionV3-LSTM architecture, we achieved a testing accuracy of 96.48% and a training accuracy of 99.98%.
Parkinson's Detection: Various machine learning models were trained on EEG-derived sensor data, with KNN achieving the highest accuracy of 93.81%.
Django Application: A user-friendly Django web application was developed to facilitate the practical use of these models in clinical settings.
Introduction
Alzheimer's Disease: A neurodegenerative disorder that impacts memory and cognitive function. Early detection is critical for slowing disease progression.
Parkinson's Disease: A disorder characterized by motor dysfunction, where early diagnosis can significantly improve patient management.
Methodology
Data Collection: EEG data classified into various stages of Alzheimer's and Parkinson's. Preprocessing steps included image scaling, normalization, and data augmentation.
Model Architecture: InceptionV3 with LSTM layers for Alzheimer's and multiple classifiers (Logistic Regression, LDA, KNN, Decision Trees) for Parkinson's.
Training Process: The models were trained using cross-validation and evaluated on multiple metrics including accuracy, MCC, and confusion matrices.
Results
Alzheimer's Detection: The InceptionV3-LSTM model demonstrated high efficacy in early detection, particularly in identifying mild cognitive impairments.
Parkinson's Detection: The KNN model outperformed others, making it a robust tool for early diagnosis.
Conclusion
This research highlights the potential of EEG data and machine learning models in the early detection of neurological disorders. The developed models and application can be valuable tools for clinicians in improving patient outcomes.
Future Research
Expanding datasets to include more diverse populations and exploring advanced architectures such as RNNs and transformers for enhanced accuracy.
