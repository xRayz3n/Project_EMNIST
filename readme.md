# AI-Powered Anti-Robot Verification

This web application integrates a state-of-the-art AI model trained using PyTorch and the EMNIST dataset, with deployment facilitated through ONNX for web compatibility. It serves as a unique approach to verify human users, replacing traditional CAPTCHAs with an interactive and AI-powered method.

## Overview

The core idea of this application is simple yet innovative: users are required to write the phrase "I'm not a robot" on a digital canvas. This user-generated input is then analyzed by our trained AI model to determine its authenticity, thereby confirming the user is indeed human.

### Key Features

- **AI Verification**: Leverages a machine learning model trained on the EMNIST dataset.
- **User-Friendly Interface**: Easy-to-use canvas for writing the verification phrase.
- **PyTorch and ONNX Integration**: Combines the power of PyTorch's machine learning capabilities with ONNX's web deployment efficiencies.

## How It Works

1. **User Interaction**: Users are prompted to write "I'm not a robot" on the canvas provided on the webpage.
2. **AI Analysis**: Once submitted, the AI model analyzes the handwriting style and pattern.
3. **Verification Outcome**: The model determines whether the writing is human-generated, confirming or denying the user's authenticity.

## Technologies Used

- **PyTorch**: For training the AI model using the EMNIST dataset.
- **EMNIST Dataset**: A comprehensive dataset of handwritten characters and digits.
- **ONNX (Open Neural Network Exchange)**: For efficient deployment of the AI model on the web platform.
- **JavaScript/HTML/CSS**: For frontend development and user interface design.
