# Age-Prediction-with-Tensorflow
Age Prediction model using the TensorFlow Estimator API.

Dataset used in this project, can be found at
https://datahack.analyticsvidhya.com/contest/practice-problem-age-detection/#About

Recommended project hierarchy is as below: (view this in raw format)
Age Prediction
  -encodings (encodings of the images and labeles will be stored in this folder)
  -result (csv file with predicted result will be stored here)
  -tensorflow-models (this folder will contain the stored model, weights, and events)
  -train (train images and train.csv)
  -test (test images and test.csv)
  -AgePred.py (main file)
 

AgePred.py loads the data and creates and stores the embedding of the images for reusability.
In model_definition() function you can play around and try different model architectures.
For model training and evaluation, I have used TensorFlow Estimator API. Estimator API gives
flexibility in providing training and validation input to the function along with its 
rich functionality of model training and evaluation. Model performance can be visualized 
in TensorBoard by giving "logdir" path to "temsorflow-models". 
For ex: tensorboard --logdir path/tensorflow-models.

