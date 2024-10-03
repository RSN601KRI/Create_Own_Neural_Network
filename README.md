# Create Own Image Classifier

## Overview

The **Create Own Image Classifier** project is designed to help users build their custom image classification models using deep learning techniques. This project leverages pre-trained models and transfer learning, making it easier and more efficient for users to create a robust image classifier without requiring extensive knowledge in machine learning.

### Project Features

- **Custom Model Training**: Users can train their own image classifiers using their dataset.
- **Pre-trained Models**: Utilize popular pre-trained models such as ResNet, VGG, and others for faster training and better accuracy.
- **User-Friendly Interface**: Simple command-line interface for executing model training and predictions.
- **GPU Support**: Option to leverage GPU for faster training times, if available.
- **Image Preprocessing**: Built-in functions to preprocess images, ensuring compatibility with the model requirements.
- **Label Mapping**: Easy mapping of category names from a JSON file for interpreting model predictions.
- **Model Saving and Loading**: Save trained models for future use and load them easily for predictions.

## Getting Started

### Prerequisites

- Python 3.x
- Pip (Python package installer)
- Required Python libraries: `torch`, `torchvision`, `PIL`, `json`, `collections`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/RSN601KRI/Create_Own_Image_Classifier.git
   cd Create_Own_Image_Classifier
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Training the Model**:
   To train your custom image classifier, run the following command:

   ```bash
   python train.py --data_dir <path_to_training_data> --save_dir <path_to_save_model>
   ```

2. **Predicting Images**:
   To make predictions on new images, use the following command:

   ```bash
   python predict.py <image_path> <checkpoint_path> --top_k <number_of_top_predictions> --category_names <path_to_category_names.json> --gpu
   ```

   - Replace `<image_path>` with the path to the image you want to classify.
   - Replace `<checkpoint_path>` with the path to the saved model checkpoint.
   - The `--top_k` argument specifies the number of top predictions you want to see.
   - The `--category_names` argument should point to a JSON file that maps class indices to names.

### Example

```bash
# Example to predict an image
python predict.py flower_data/test/image_05100.jpg saved_models/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

## About the Project

The **Create Own Image Classifier** project aims to empower users to leverage the power of deep learning for image classification tasks. By simplifying the model training process and providing easy-to-use features, this project serves as an excellent starting point for anyone looking to delve into the field of computer vision.

This project is perfect for:

- Students and beginners wanting to learn about image classification.
- Developers looking to integrate custom image classification into their applications.
- Researchers exploring new datasets and model architectures.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the contributors of the PyTorch community for their excellent resources and libraries.
- Inspiration from various open-source image classification projects.
