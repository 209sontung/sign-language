![Logo](/assets/sign_logo_straight.png)

![GitHub](https://img.shields.io/github/license/209sontung/sign-language?style=flat-square)
![GitHub top language](https://img.shields.io/github/languages/top/209sontung/sign-language?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/209sontung/sign-language?color=yellow&style=flat-square)
![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/209sontung/sign-language?color=orange&style=flat-square)

### Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Directory Structure](#directory)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)


# Real-time Sign Language Gesture Recognition Using 1DCNN + Transformers on MediaPipe landmarks

## <a name="overview"></a> Overview
Sign language is an important means of communication for individuals with hearing impairments. This project aims to build a real-time sign language gesture recognition system using deep learning techniques. The system utilizes 1D Convolutional Neural Networks (1DCNN) and Transformers to recognize sign language gestures based on the hand landmarks extracted from MediaPipe.

## <a name="prerequisites"></a> Prerequisites

- Python 3.9
- NumPy
- Pandas
- Matplotlib
- OpenCV
- MediaPipe
- TensorFlow

## <a name="installation"></a> Installation
1. Clone this repository to your local machine using either the HTTPS or SSH link provided on the repository's GitHub page. You can use the following command to clone the repository via HTTPS:

```bash
git clone https://github.com/209sontung/sign-language
```

2. Once the repository is cloned, navigate to the root directory of the project:

```bash
cd sign-language
```

3. It is recommended to create a virtual environment to isolate the dependencies of this project. You can create a virtual environment using venv module. Run the following command to create a virtual environment named "venv":

```bash
python3 -m venv sign-language
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

You have successfully installed the system and are ready to use it for real-time sign language gesture recognition. Please refer to the [Usage](#usage) section in the `README.md` for instructions on how to run and utilize the system.

## <a name="usage"></a> Usage
To use the sign language gesture recognition system, follow these steps:

1. Ensure that you have installed all the required dependencies (see [Installation](#installation)).

2. Run the main.py file, which contains the main script for real-time gesture recognition.

```bash
python main.py
```

3. The system will start capturing your hand gestures using the webcam and display the recognized gestures in real-time.

## <a name="models"></a> Models
The repository includes pre-trained models for sign language gesture recognition. The following models are available in the models directory:

- islr-fp16-192-8-seed42-fold0-best.h5: Best model weights for fold 0.
- islr-fp16-192-8-seed42-fold0-last.h5: Last model weights for fold 0.
- islr-fp16-192-8-seed_all42-foldall-last.h5: Last model weights for all folds.

## <a name="directory"></a> Directory Structure
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
├─ Pipfile
├─ Pipfile.lock
└─ src
   ├─ backbone.py
   ├─ config.py
   ├─ landmarks_extraction.py
   ├─ sign_to_prediction_index_map.json
   └─ utils.py
```

## <a name="contributing"></a> Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## <a name="acknowledgments"></a> Acknowledgments
Sincere gratitude is extended to @hoyso48 for providing the idea and initial implementation of the backbone model.

## <a name="license"></a> License
This project is licensed under the MIT License. Feel free to use and modify the code as per the terms of the license.
