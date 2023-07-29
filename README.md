# Face Mask Detection


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Tools](#Tools)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Project Demo](#ProjectDemo)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Face Mask Detection is a computer vision project aimed at detecting whether a person is wearing a face mask or not. The project utilizes deep learning techniques to classify faces into two categories: "With Mask" and "Without Mask." The motivation behind this project is to help enforce mask-wearing policies in public places and contribute to public health and safety during pandemics like COVID-19.

## Features

- Real-time face mask detection from images or live video streams.
- High accuracy due to the deep learning model used.
- Easy-to-use interface for both developers and end-users.
- Streamlit WebApp, Simple and accurate.

## Tools
<img width="587" alt="Screenshot 2023-07-29 152144" src="https://github.com/ankitrajput77/FaceMaskDetection/assets/113281225/aa161e65-9041-4c1b-b5b5-55b9bf531d53">
<img width="439" alt="Screenshot 2023-07-29 152152" src="https://github.com/ankitrajput77/FaceMaskDetection/assets/113281225/44be9c76-1426-419f-99fa-6af51c6df663">

## Installation

To set up the Face Mask Detection project, follow these steps:

1. Clone the repository:
```
git clone https://github.com/ankitrajput77/FaceMaskDetection.git
```
2. Navigate to the project directory:
```
cd FaceMaskDetection
```
3. Open Ternimal as Conda Base environment and Environment create
```
conda create -p env python==3.10.0
```
```
conda activate env/
```
4. Install dependencies
```
pip install -r requirements.txt
```
5. Run the WebApp
```
streamlit run app.py
``` 
## Dataset

https://www.kaggle.com/datasets/ahmedabdelraouf/face-datasets

## Model

The deep learning model used for this project is a custom convolutional neural network (CNN) architecture. It was trained on the dataset mentioned above to detect face masks accurately. The model is implemented using popular deep learning frameworks like TensorFlow/Keras.

```
├──  artifacts                    - here's the models stored.
│    └── Mask_detection_model.h5  
│    └── Face Models
│      └── res10_300x300_ssd_iter_140000.caffemodel
│      └── deploy.prototxt
```
## Results
The model achieved an accuracy of 96% on the test dataset. The performance may vary depending on the dataset and the quality of the images used for detection.

<img width="573" alt="Screenshot 2023-07-29 151541" src="https://github.com/ankitrajput77/FaceMaskDetection/assets/113281225/4335d6ac-66a6-452c-a8c4-1ac59237215a">
<img width="536" alt="Screenshot 2023-07-29 151552" src="https://github.com/ankitrajput77/FaceMaskDetection/assets/113281225/d267d5d0-5ec5-4b5d-a861-be3cf9bd0f4a">

## ProjectDemo
https://github.com/ankitrajput77/FaceMaskDetection/assets/113281225/094d1030-83f1-431c-a005-40c9af297668

## Contributing
Contributions to this project are welcome. If you find any issues or want to enhance the functionality, feel free to open a pull request. Please make sure to follow the coding conventions and provide detailed information about the changes.

## License
This project is licensed under the MIT License.


