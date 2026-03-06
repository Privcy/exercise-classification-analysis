# Exercise Form Analysis using Pose Estimation and ROCKET

## Project Overview
This repository contains the code and methodology for analyzing and classifying exercise forms—specifically squats and deadlifts. The project leverages advanced human pose estimation models to extract skeletal keypoints from video data, which are then processed as multivariate time-series data and classified using the ROCKET (Radom Convolutional Kernel Transform) algorithm.

## Tech Stack
- **Pose Estimation:** OpenPose, RTMPose, MediaPipe
- **Time-Series Classification:** ROCKET (via 'sktime')
- **Data Processing:** python, pandas, numpy, opencv
- **Environment:** Pycharm (Jupyter notebooks works too)


## Setup and Installation
  1. cloning: _git clone [https://github.com/Privcy/exercise-classification-analysis.git](https://github.com/Privcy/exercise-classification-analysis.git)
cd exercise-classification-analysis_

  2. create virtual environment: _python -m venv venv_

  3. activating the enviroment: _venv\Scripts\activate_
  (however in my case i used pycharm so _conda activate BodyMTS_ was mine)

  4. installing dependencies: _pip install -r requirements.txt_

## Methodology
  1. Data Collection: video capture of subjects performing target exercises (shot on iPhone 12 pro max).
  2. Pose Extraction: passing video frames through the pose estimation models to extract coordinates of keypoints over time.
  3. Data Transformation: converting coordinates into .ts (time series) format for modeling.
  4. Classification: training a ROCKET classifier to identify form deviations, movement quality, or specific exercise phases.

## Acknowledgements and References
- This project integrates foundational concepts and reference code from the BodyMTS_2021 repository.
- * **Base paper:** Singh, A., Bevilacqua, A., Nguyen, T. L., Hu, F., McGuinness, K., O'Reilly, M., Whelan, D., Caulfield, B., & Ifrim, G. (2022). Fast and robust video-based exercise classification via body pose tracking and scalable multivariate time series classifiers. arXiv. https://doi.org/10.48550/arXiv.2210.00507
