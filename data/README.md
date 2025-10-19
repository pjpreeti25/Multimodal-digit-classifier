# Data Directory

**⚠️ IMPORTANT: Due to data privacy and confidentiality requirements, the actual dataset files are not included in this repository.**

## Required Files
To run this project, you'll need to obtain the following data files and place them in this directory:

- `x_train_wr.npy` - Training image data (shape: [N, 28*28])
- `x_train_sp.npy` - Training audio data (shape: [N, 507])
- `x_test_wr.npy` - Test image data (shape: [M, 28*28])
- `x_test_sp.npy` - Test audio data (shape: [M, 507])
- `y_train.csv` - Training labels with 'label' column

## Data Description
- **Images**: 28×28 grayscale handwritten digits
- **Audio**: 507-dimensional audio features corresponding to each digit
- **Labels**: Digit classes 0-9

## Usage
Once you have the data files, place them in this directory and run the preprocessing pipeline:

```python
from src.preprocess import preprocess_data
data = preprocess_data(data_path='data/')
```

## Data Privacy Notice
This project was developed using proprietary/confidential data that cannot be shared publicly. The code structure and methodology are demonstrated through the implementation, but actual data files must be obtained separately and placed in this directory to reproduce the results.
