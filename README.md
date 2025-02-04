# CRNN for Text Recognition

This repository provides an implementation of a Convolutional Recurrent Neural Network (CRNN) for sequence modeling tasks, such as text recognition. The model leverages a Convolutional Neural Network (CNN) for feature extraction and a Long Short-Term Memory (LSTM) network for sequence modeling, making it well-suited for tasks that require handling variable-length sequences (e.g., OCR with CTC loss).

## Features

- **CNN Feature Extractor**: A series of convolutional and pooling layers to extract high-level features from input images.
- **LSTM Sequence Modeler**: An LSTM processes sequential features extracted from the CNN.
- **CTC Loss Compatibility**: The network output includes an extra "blank" label for compatibility with Connectionist Temporal Classification (CTC) loss.
- **Checkpointing**: Automatically saves model checkpoints after each epoch for resuming training later.
- **Customizable Architecture**: Parameters like LSTM hidden size, number of layers, and number of classes can be easily adjusted.

## Requirements

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)

Additional dependencies (e.g., NumPy) might be required based on your dataset and training pipeline.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/CRNN-Text-Recognition.git
   cd CRNN-Text-Recognition
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install torch torchvision
   # If using additional dependencies, list them here.
   ```

## Usage

### Dataset & Data Loaders

- **Data Preparation:**  
  Make sure you have defined your data loaders (`train_loader` and `test_loader`) to feed training and testing data into the model.
- **Character Mapping:**  
  Define the `char_to_index` mapping that converts characters to indices. The length of this mapping determines the number of classes for the model.

### Model Training

The main training loop is provided in the script. Key points include:

- **Model Initialization:**  
  The model is defined as `CRNN(num_classes+1)`, where the extra class is reserved for the blank token used in CTC loss.

- **Checkpoint Loading:**  
  To resume training from a saved checkpoint, set the `load` variable to `True`. The checkpoint is then loaded from `checkpoint.pth`.

- **Running Training:**  
  The training loop runs for a preset number of epochs (default is 100). After each epoch, training loss, test loss, and test accuracy are printed, and a checkpoint is saved.

To start training, simply run:

```bash
python main.py
```
