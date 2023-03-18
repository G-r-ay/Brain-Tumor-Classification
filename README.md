# Brain Tumor Classification Project
This project is focused on building a machine learning model that can accurately classify brain tumors based on MRI scans. The model was built using PyTorch, a popular deep learning library, and achieved a testing accuracy of 0.8777.

## Dataset
The dataset used in this project is the [Brain Tumor Classification (MRI) dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which consists of MRI scans of the brain with and without tumors. The dataset contains 253 images of non-tumor scans and 253 images of tumor scans.

## Model
The model used in this project is a convolutional neural network (CNN), which is a type of deep learning model commonly used for image classification tasks. The model architecture consists of several convolutional layers followed by max pooling layers, and a fully connected layer at the end. The model was trained using PyTorch and achieved a testing accuracy of 0.8777.

## Repository Contents
The repository contains the following files:

brain_tumor_classification.ipynb: A Jupyter notebook containing the code used to build and train the model.
model_state_dict.pth: The state dictionary of the trained model.
README.md: This file.

Usage
To use the trained model to classify brain tumors, you can load the state dictionary using PyTorch and apply the model to new MRI scans. Here's an example code snippet:

``` python
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the model state dictionary
model_state_dict = torch.load('model_state_dict.pth')

# Define the model architecture

class BrainClassifier(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim):
        super(BrainClassifier,self).__init__()
        
        self.Seq1 = nn.Sequential(
            nn.Conv2d(in_channels = input_dim, out_channels=hidden_dim,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_dim, out_channels=hidden_dim,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(stride=2,kernel_size=2)
        )
        
        self.Seq2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_dim, out_channels=hidden_dim,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_dim, out_channels=hidden_dim,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(stride=2,kernel_size=2)
        )
        self.Seq3 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_dim, out_channels=hidden_dim,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_dim, out_channels=hidden_dim,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(stride=2,kernel_size=2)
        )
        self.Seq4= nn.Sequential(
            nn.Conv2d(in_channels = hidden_dim, out_channels=hidden_dim,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_dim, out_channels=hidden_dim,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(stride=2,kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_dim*4*4,out_features=output_dim)
        )
        
    def forward(self, x :torch.Tensor):
        
        x = self.Seq1(x)
        print(x.shape)
        x = self.Seq2(x)
        print(x.size())
        x = self.Seq3(x)
        print(x.size())
        x = self.Seq4(x)
        print(x.size())
        x = self.classifier(x)
        
        return x
        
        
model = BrainClassifier(input_dim=3,output_dim=4, hidden_dim =128)

# Load the state dictionary into the model
model.load_state_dict(model_state_dict)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image and apply the transformations
image = Image.open('path/to/image.jpg')
image = transform(image)

# Apply the model to the image
output = model(image.unsqueeze(0))
```

### Conclusion
This project demonstrates the use of deep learning techniques for brain tumor classification based on MRI scans. The trained model achieved a testing accuracy of 0.8777 and can be used to classify new MRI scans. The model_state_dict.pth file is included in the repository for easy access to the trained model.
