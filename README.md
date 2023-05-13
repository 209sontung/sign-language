## Table of Contents
- Overview
- Prerequisites
- Installation
- Usage
- Models
- Directory Structure
- Contributing
- License


# Real-time Sign Language Gesture Recognition
This is a GitHub repository for a real-time sign language gesture recognition system using 1DCNN + Transformers on MediaPipe landmarks. The system is capable of recognizing sign language gestures in real-time based on the hand landmarks extracted from MediaPipe.

## Overview
Sign language is an important means of communication for individuals with hearing impairments. This project aims to build a real-time sign language gesture recognition system using deep learning techniques. The system utilizes 1D Convolutional Neural Networks (1DCNN) and Transformers to recognize sign language gestures based on the hand landmarks extracted from MediaPipe.

## Prerequisites:

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- OpenCV
- MediaPipe
- TensorFlow

## Installation
1. Clone this repository to your local machine using either the HTTPS or SSH link provided on the repository's GitHub page. You can use the following command to clone the repository via HTTPS:

```bash
git clone https://github.com/209sontung/sign-language
```

2. Once the repository is cloned, navigate to the root directory of the project:

```bash
cd real-time-sign-language-gesture-recognition
```

3. It is recommended to create a virtual environment to isolate the dependencies of this project. You can create a virtual environment using venv module. Run the following command to create a virtual environment named "venv":

```bash
python3 -m venv venv
```

4. Activate the virtual environment. The activation steps depend on the operating system you're using:

- For Windows:
```bash
venv\Scripts\activate
```
- For macOS/Linux:
```bash
source venv/bin/activate
```

5. Now, you can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

6. Once the installation is complete, you're ready to use the real-time sign language gesture recognition system.

**Note:** Make sure you have a webcam connected to your machine or available on your device for capturing hand gestures.

You have successfully installed the system and are ready to use it for real-time sign language gesture recognition. Please refer to the Usage section in the README for instructions on how to run and utilize the system.

## Models
The repository includes pre-trained models for sign language gesture recognition. The following models are available in the models directory:

- islr-fp16-192-8-seed42-fold0-best.h5: Best model weights for fold 0.
- islr-fp16-192-8-seed42-fold0-last.h5: Last model weights for fold 0.
- islr-fp16-192-8-seed_all42-foldall-last.h5: Last model weights for all folds.

## Directory Structure
The directory structure of this repository is as follows:

```bash

├─ .gitignore
├─ LICENSE
├─ README.md
├─ main.py
├─ models
│  ├─ islr-fp16-192-8-seed42-fold0-best.h5
│  ├─ islr-fp16-192-8-seed42-fold0-last.h5
│  └─ islr-fp16-192-8-seed_all42-foldall-last.h5
├─ requirements.txt
└─ src
   ├─ backbone.py
   ├─ config.py
   ├─ landmarks_extraction.py
   ├─ sign_to_prediction_index_map.json
   └─ utils.py
```

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. Feel free to use and modify the code as per the terms of the license.
