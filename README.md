# Road Extraction from Satellite Images

## Introduction
Extracting roads from satellite images is a crucial step toward smarter cities and efficient disaster response. This project focuses on training a deep learning model to identify and segment roads from satellite images.

## Methodology

### Dataset Preparation
The dataset comprises satellite images and their corresponding road masks:
- **Training Data:** Includes satellite images and their masks.
- **Test Data:** Contains only images for evaluating the model’s predictions.

We applied different types of data augmentation:
- **Type 1:** No augmentation.
- **Type 2:** Applied the following transformations:
  ```python
  transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  ```
- **Type 3:** Applied only normalization:
  ```python
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ```

### Model Architecture
We used **DeepLabV3-ResNet101**, a state-of-the-art segmentation model, due to its robustness and accuracy.
- **Feature Extraction with ResNet-101:** Extracts high-level features using deep convolutional layers.
- **Atrous Spatial Pyramid Pooling (ASPP):** Captures multi-scale contextual information for better segmentation.
- **Upsampling:** Ensures that the segmentation output maintains high spatial resolution and accuracy.

### Training Strategy
- **Train-Validation Split:** 80% training, 20% validation.
- **Optimization Techniques:**
  - **Learning Rate Scheduling:** Dynamically adjusts learning rate for stability.
  - **Weight Decay:** Prevents overfitting by penalizing large weights.
  - **Cross-Entropy Loss:** Measures performance by comparing predictions with true labels.
  - **Adam Optimizer:** Adaptive learning rate optimization.
  - **StepLR Scheduler:** Adjusts learning rate at intervals.
  - **Epochs:** Multiple iterations for better learning.

### Evaluation Process
The **F1-score** was used as the primary evaluation metric, balancing precision and recall to ensure accurate road detection.

## Results

### Quantitative Metrics

#### Training and Validation Performance
```
Learning Rate | Weight Decay | Batch Size | Epochs | Data Augmentation | Training Accuracy | Validation Accuracy | Optimizer | Loss Function
-------------------------------------------------------------------------------------------------------------
0.0002       | 0.0005       | 8          | 35     | Type 2             | 96.31              | 0.8357               | Adam      | Cross-Entropy
0.0002       | 0.0005       | 64         | 60     | Type 2             | 94.91              | 0.8103               | Adam      | Cross-Entropy
0.0002       | 0.0005       | 32         | 50     | Type 2             | 95.78              | 0.7969               | Adam      | Cross-Entropy
0.0001       | 0.001        | 16         | 40     | Type 2             | 94.99              | 0.8155               | Adam      | Cross-Entropy
0.0002       | 0.0005       | 16         | 40     | Type 2 (No Resize) | 99.00              | 0.91                 | Adam      | Cross-Entropy
0.0002       | 0.0005       | 16         | 40     | Type 1             | 100.00             | 0.98                 | Adam      | Cross-Entropy
0.0002       | 0.0005       | 8          | 35     | Type 1             | 96.69              | 0.9247               | Adam      | Cross-Entropy
0.0002       | 0.0005       | 32         | 55     | Type 3             | 92.45              | 0.8940               | Step (0.01) | Cross-Entropy
0.0002       | 0.0005       | 8          | 35     | Type 3             | 95.83              | 0.9178               | Step (0.1) | Cross-Entropy
0.0002       | 0.0005       | 8          | 35     | Type 1             | 94.00              | 0.90                 | Adam      | Cross-Entropy
0.0003       | 0.0001       | 8          | 35     | Type 3             | 96.87              | 0.9300               | Adam      | Cross-Entropy
```

### Qualitative Analysis
The model successfully segmented roads in most test images, but some challenges remained:
- **Misclassifications:** Similar features like parking lots confused the model.
- **Occlusions:** Roads obscured by trees or buildings were harder to detect.

## Conclusion
This project demonstrated the effectiveness of **DeepLabV3-ResNet101** for road extraction from satellite images, achieving a strong **F1-score**. Key insights:
- **Data augmentation initially decreased performance**, so we relied on normalization.
- **DeepLabV3's flexible input handling** simplified the workflow.
- **Challenges remain with occlusions and similar textures** (e.g., parking lots).
- **Further improvements:** More training images and alternative loss functions (e.g., Dice Loss) could enhance accuracy.

While other models might have offered better performance, we chose **DeepLabV3** for its initial success. Due to time constraints and heavy coursework, we couldn't explore all possible alternatives.

---

🚀 **Future Work:** Exploring alternative models, larger datasets, and improved loss functions to enhance segmentation performance.
