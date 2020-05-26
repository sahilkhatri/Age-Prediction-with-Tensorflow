# import required libraries
import numpy as np 
import pandas as pd
import cv2
cv2.__version__
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from keras.utils.np_utils import to_categorical
import pickle
from keras.models import model_from_yaml

"""
    class AgePrediction containes all the functionality for loading data, storing encodings, loading encodings,
    input functions for estimator api, train and eval function, along with test function.
"""
class AgePrediction:
    def __init__(self):        
        
        # load train.csv from the directory specified
        os.chdir('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/train/Train')
        
        # load the csv file in the pandas dataframe
        self.data = pd.read_csv('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/train/train.csv')  
        
        # if encodings does not exist, then call store_encodings function
        # else call load_encodings function
        os.chdir('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/encodings')
        if len(os.listdir())==0:
            X_data, labels = self.store_encodings()
        else:
            X_data, labels = self.load_encodings()
        
        # call data_preparation function which returns the splitted data
        x_train, y_train, x_test, y_test, x_valid, y_valid = self.data_preparation(X_data, labels)
    
        # call model_definition function to define the model architecture
        model = self.model_definition()
        
        # define the output path to store the trained models into
        OUTDIR = '/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/tensorflow-models/'
        
        # create writer object of tf.summary for plotting result into tensorboard
        self.writer = tf.summary.create_file_writer(OUTDIR)
        
        # call train_eval_model for training the model
        self.train_eval_model(model, x_train, y_train, x_valid, y_valid)
        
        # call predict_model for making predictions on the test data
        self.predict_model(model, x_test, y_test)
        
        # call test_result for storing the predicted result as the required format
        # actual test data is used here on which we need to make predictions
        self.test_result(model)
        
    """
        store_encodings functions loads the images from the image id given in the "data" dataframe.
        While loading the images are resized into 32 x 32.
        Class of the image (i.e. YOUNG, MIDDLE, OLD) are converted to categorical labels.
        Both these images and categorical labels are then returned as well as stored in the pickle file.
    """                
    def store_encodings(self):
        X_data = []
        for i in self.data['ID']:
            X_data.append(cv2.resize(cv2.imread('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/train/Train/'+i, cv2.IMREAD_COLOR), (32,32), interpolation = cv2.INTER_CUBIC))   
        classes = []
        for i in self.data['Class']:
            if i=='YOUNG':
                classes.append(0)
            if i=='MIDDLE':
                classes.append(1)
            if i=='OLD':
                classes.append(2)           
        classes[:10]            
        categorical_labels = to_categorical(classes, num_classes=3)
        print(categorical_labels[:10]) 
        pickle.dump(X_data, open('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/encodings/images_32_cv2','wb'))
        pickle.dump(categorical_labels, open('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/encodings/age_32_cv2','wb'))
        return X_data, categorical_labels
    
    """
        load_encodings function loads the images and categorical labels from the pickle file and returns them.
    """
        
    def load_encodings(self):
        X_data = pickle.load(open('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/encodings/images_32_cv2','rb'))
        categorical_labels = pickle.load(open('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/encodings/age_32_cv2','rb'))
                
        return X_data, categorical_labels
    
    """
        data_prepapration function takes input the images and categorical labels.
        It normalizes the images by dividing each value with 255.
        And also divides the data into training, validation, and testing.
        Finally returns the numpy array of splitted data.
    """
    def data_preparation(self, X_data, labels):
        X_data = np.squeeze(X_data)
        #X_data.shape
        X_data = X_data.astype('float32')
        X_data /= 255
        (x_train, y_train), (x_test, y_test) = (X_data[4000:],labels[4000:]) , (X_data[:4000], labels[:4000])
        (x_valid , y_valid) = (x_test[:3500], y_test[:3500])
        (x_test, y_test) = (x_test[3500:], y_test[3500:])

        return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), np.asarray(x_valid), np.asarray(y_valid)

    """
        make_train_input_fu takes the training data and number of epochs as input parameter.
        it returns the numpy_input_fn of the estimator api. This function helps to load the
        training data at the time of model training.
    """
    def make_train_input_fn(self, x_train, y_train, num_epochs):
        return tf.compat.v1.estimator.inputs.numpy_input_fn(
                x = x_train,
                y = y_train,
                batch_size = 64,
                num_epochs = num_epochs,
                shuffle = True,
                queue_capacity = 500
                )

    """
        make_valid_input_fu takes the validation data as input parameter.
        it returns the numpy_input_fn of the estimator api. This function helps to load the
        validation data at the time of model training.
    """
    def make_valid_input_fn(self, x_valid, y_valid):
        return tf.compat.v1.estimator.inputs.numpy_input_fn(
                x = x_valid,
                y = y_valid,
                batch_size = 64,
                shuffle = False,
                queue_capacity = 500
                )
      
    """
        make_test_input_fu takes the test data input parameter.
        it returns the numpy_input_fn of the estimator api. This function helps to load the
        test data at the time of model testing/prediction.
    """
    def make_test_input_fn(self, x_test):
        return tf.compat.v1.estimator.inputs.numpy_input_fn(
                x = x_test,
                y = None,
                batch_size = 64,
                shuffle = False,
                queue_capacity = 500
                )

    """
        model_definition funciton defines the architecture of the model.
        Model architecture:
            Sequential
            Convolution2d       : 64, 3x3, relu
            BatchNormalization
            MaxPooling2D        : 2x2
            Convolution2d       : 32, 3x3, relu
            BatchNormalization
            MaxPooling2D        : 2x2
            Convolution2d       : 16, 2x2, relu
            BatchNormalization
            MaxPooling2D        : 2x2
            Flatten
            Dense               : 256, relu
            Dense               : 64, relu
            Dense               : 3, softmax
            optimizer           : adam
            loss                : categorical_crossentropy
        This function returns the defined model.
    """
    def model_definition(self):
        model = tf.keras.models.Sequential([
                tf.keras.layers.Convolution2D(filters=64, kernel_size=5, padding='same', activation='relu', input_shape=(32,32,3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=3),
#                tf.keras.layers.Dropout(0.3),
        
                tf.keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                tf.keras.layers.Dropout(0.3),
        
                tf.keras.layers.Convolution2D(filters=16, kernel_size=2, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=2),
#                tf.keras.layers.Dropout(0.3),
        
        
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     optimizer='adam',
                     metrics=['accuracy'])
        model.summary()
        
        return model
    
    """
        train_eval_model takes model, training data, and validation data as input parameters.
        Here we define the estimator runconfig (output dir to store model, seed, and save sumamry steps).
        Then we convert the keras model to the estimator using model_to_estimator functionality of the 
        estimator api.
        
        We define the train_spec which takes the input from the make_train_input_fn (indirectly from the 
        numpy_input_fn of the estimator api).
        We define the eval_spec which takes the input from the make_valid_input_fn (indirectly from the 
        numpy_input_fn of the estimator api).
        
        Using train_and_evaluate function of the estimator api which takes the estimator object and train
        and eval spec as input, and starts the training of the model.
        
        Instead of creating the train_spec and eval_spec we can also directly train and evaluate model by 
        using the train function and evaluate function of the estimator object respectively. 
    """
        #check tf.config on tensorflow
    def train_eval_model(self, model, x_train, y_train, x_valid, y_valid):
        OUTDIR = '/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/tensorflow-models/'
        config = tf.estimator.RunConfig(model_dir = OUTDIR, tf_random_seed=0,save_summary_steps=10)
        keras_estimator = tf.keras.estimator.model_to_estimator(
                keras_model = model, config = config)
        
#        train_spec = tf.estimator.TrainSpec(input_fn = self.make_train_input_fn(x_train, y_train, 30),
#                                            max_steps = 4000)

        train_spec = tf.estimator.TrainSpec(input_fn = self.make_train_input_fn(x_train, y_train, 90))

        
        eval_spec = tf.estimator.EvalSpec(input_fn = self.make_valid_input_fn(x_valid, y_valid),
                                          steps = None,
                                          start_delay_secs = 1,
                                          throttle_secs = 10)
    
        result = tf.estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)        
        
        print(result)
#        keras_estimator.train(input_fn = self.make_train_input_fn(x_train,y_train, num_epochs = 30))
        
#        eval_result = keras_estimator.evaluate(input_fn = self.make_valid_input_fn(x_valid,y_valid))
#        print('Eval results: {}'.format(eval_result))
        
#        with self.writer.as_default():
#            tf.summary.scalar('validation loss', eval_result['loss'], step = eval_result['global_step'])
#            tf.summary.scalar('validation accuracy', eval_result['accuracy'], step = eval_result['global_step'])


    """
        preict_model takes model definition as input along with the test data.
        It predicts the result using the predict function of the estimator object.
        Based on the results obtained from the prediction, the images are plotted
        along with their actual and predicted label value.
        
        predict function of the estimator api returns the generator object, so
        the results are in dictionary format, which needs to be manipulated to 
        list so that actual label values (YOUNG, MIDDLE, OLD) can be extracted.
    """
    def predict_model(self, model, x_test, y_test):
        OUTDIR = '/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/tensorflow-models/'
        keras_estimator = tf.keras.estimator.model_to_estimator(
                keras_model = model, model_dir = OUTDIR)

        pred_result = keras_estimator.predict(input_fn = self.make_test_input_fn(x_test))
#        print('Prediction results: {}'.format(pred_result))
        
        labels =["YOUNG",  # index 0
                 "MIDDLE",  # index 1 
                 "OLD"   # index 2
                 ]
        
        pred_result = list(pred_result)
        
        figure = plt.figure(figsize=(20, 8))
        for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
            ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(x_test[index]))
            temp = list(pred_result[index].values())
            print(temp)
            predict_index = np.argmax(temp)
            print(predict_index)
            true_index = np.argmax(y_test[index])
            ax.set_title("{} ({})".format(labels[predict_index], 
                                  labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
        plt.show()


    """
        test_result function takes model definition as the input. These function 
        load the test.csv into a dataframe, and load the images using the 'ID' value
        of the dataframe. These images are then normalized by dividing the values 
        with 255. Again using the estimator object we make prediction on the test data
        that we just loaded. Based on the results obtained, the labels are converted
        to its actual values (i.e. YOUNG, MIDDLE, OLD). After that these predicted labels
        are stored in the csv file as per the required format (i.e. image id corresponding
        to the class predicted).
    """

    def test_result(self, model):
        os.chdir('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/test/Test')
        test_data = pd.read_csv('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/test/test.csv')       
        test_images = []
        for i in test_data['ID']:
            test_images.append(cv2.resize(cv2.imread('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/test/Test/'+i, cv2.IMREAD_COLOR), (32,32), interpolation = cv2.INTER_CUBIC))
        test_images = np.squeeze(test_images)
        test_images = test_images.astype('float32')
        test_images /= 255
        
        OUTDIR = '/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/tensorflow-models/'
        keras_estimator = tf.keras.estimator.model_to_estimator(
                keras_model = model, model_dir = OUTDIR)

        pred_test = keras_estimator.predict(input_fn = self.make_test_input_fn(test_images))
        labels =["YOUNG",  # index 0
                 "MIDDLE",  # index 1 
                 "OLD"    # index 2
                 ]        
        temp = []
        pred_test = list(pred_test)
        for i in range(len(pred_test)):
            temp_list = list(pred_test[i].values())
            temp.append(labels[np.argmax(temp_list)])
        test_data['Class'] = temp
        test_data1 = test_data.reindex(columns=["Class","ID"])
        test_data1.to_csv('/home/sahil/silvertouch AIML/WFH evaluation/Age Prediction/result/test_output.csv',index=False)
        
        
        
age = AgePrediction()   # class instantiation
# tensorboard --logdir tensorflow-models