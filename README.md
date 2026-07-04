*# 🛡️ VERITAS AI - DeepFake Video Detection Using Transfer Learning*



*## 📌 Project Overview*

*A deep learning model to detect deepfake videos using transfer learning with ResNet18, achieving \*\*81.38% accuracy\*\* on the Celeb-DF dataset.*



*## 🎯 Problem Statement*

*Deepfakes are manipulated videos that can spread misinformation. This project aims to automatically detect deepfake videos using AI.*



*## 📊 Dataset*

*- \*\*Source:\*\* Celeb-DF (Kaggle)*

*- \*\*Images:\*\* 12,000 (6,000 real + 6,000 fake)*

*- \*\*Split:\*\* 8,000 Train, 2,000 Validation, 2,000 Test*



*## 🏗️ Methodology*

*1. \*\*Data Collection\*\* - Celeb-DF dataset*

*2. \*\*Preprocessing\*\* - Resize to 224x224, normalization, augmentation*

*3. \*\*Transfer Learning\*\* - ResNet18 pretrained on ImageNet*

*4. \*\*Fine-tuning\*\* - Unfreeze and train on deepfake dataset*

*5. \*\*Evaluation\*\* - Accuracy, F1 Score, Confusion Matrix, ROC Curve*



*## 🧠 Model Architecture*

*- \*\*Base Model:\*\* ResNet18 (pretrained on ImageNet)*

*- \*\*Modified:\*\* Replaced final layer for binary classification (Real/Fake)*

*- \*\*Parameters:\*\* 11.2M total, 1,026 trainable*



*## 📈 Results*



*| Metric | Score |*

*|--------|-------|*

*| \*\*Accuracy\*\* | \*\*81.38%\*\* |*

*| \*\*Precision\*\* | \*\*86.50%\*\* |*

*| \*\*Recall\*\* | \*\*74.35%\*\* |*

*| \*\*F1 Score\*\* | \*\*0.7997\*\* |*

*| \*\*ROC AUC\*\* | \*\*0.8945\*\* |*



*### Confusion Matrix*

