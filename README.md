## END TO END MACHINE LEARNING DATA SCIENCE PROJECT USING CI/CD PIPELINES

We will be implementing our end to end to end data science project we will be tackling  a regression project where we willbe tring to predict the maths score using various  indipendent variables.  
we will start by connecting to our git hub where we will create  a github repository  where we will be commiting our code , we will create our README.md file and our .gitignore. this process i will document it from end to end and we can procees to deploy it to the cloud service (AWS).
We will begin by creating tthe environment we start by the code
'''
conda create -p venv python==3.9 -y
'''
As we  can clearly se in the  above line of code we will be using python version 3.9
then we will activate the environment using 
'''
conda activate venv/
'''
form here we will need to create a   SETUP.PY FILE.

'''
from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
nanme='endtoendproject',
version='0.0.1',
author='stilinsk',
author_email='kamandesimone@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)
'''
The requirememnts.txt which contains all the dependencies looks as follows  these are the foloowing libararies we will be using for our machine learning project 

'''
pandas
numpy
scikit-learn
seaborn
matplotlib
xgboost
catboost
flask
dill
-e .
'''
## Here is a brief explanation of what the entire code looks like  and does
The `setup.py` file you provided is used to define the configuration and dependencies for your Python package. In an end-to-end data science project, the `setup.py` file is typically used when you want to distribute your project as a package that can be installed and used by others.

Here's a breakdown of the code in the `setup.py` file:

1. Import statements: The code begins by importing necessary modules and functions. In this case, it imports `find_packages` and `setup` from `setuptools` and `List` from `typing`.

2. Function definition: The code defines a function named `get_requirements` that takes a file path as input and returns a list of requirements. This function reads the contents of the specified file, removes newline characters, and removes the `-e .` requirement if present.

3. `setup()` function call: The `setup()` function is called with various arguments to configure your package. The arguments include:
   - `name`: The name of your package ('endtoendproject' in this case).
   - `version`: The version of your package ('0.0.1' in this case).
   - `author` and `author_email`: Your name and email address.
   - `packages`: The list of packages to include in your distribution. `find_packages()` is used to automatically find all packages in your project.
   - `install_requires`: The list of requirements for your package. It calls the `get_requirements()` function with the path to the `requirements.txt` file to retrieve the requirements.

4. `requirements.txt` file: The `requirements.txt` file contains a list of dependencies required by your project. In this case, the listed dependencies include 'pandas', 'numpy', 'scikit-learn', 'seaborn', 'matplotlib', 'xgboost', 'catboost', 'flask', and 'dill'. The `-e .` requirement is excluded when retrieving the requirements in the `get_requirements()` function.

By defining the `setup.py` file and listing the project dependencies in the `requirements.txt` file, you can easily distribute your project as a package and allow others to install it along with its required dependencies using package managers like pip.
after this a file will  be created looks 

We will start by creating our src folder and in int we will create a file named as __init__.py

     src folder: The src folder stands for "source" and is used to store the actual source code files of your project. It provides a dedicated location for your Python modules, packages, and other related files.

    __init__.py file: The __init__.py file is an empty file that serves as an indicator that the directory is a Python package. It can also contain initialization code or define variables or   functions that should be accessible when the package is imported.

    By creating the src folder and including an __init__.py file, you are defining a Python package structure. This structure allows you to organize your code logically and provides a clean separation between different modules and packages within your project.

data_ingestion.py: This file could contain code related to data ingestion, such as reading data from different sources (e.g., files, databases, APIs) and preprocessing it for further analysis.

data_transformation.py: This file could include code for transforming or cleaning the data. It might involve tasks like feature engineering, data normalization, handling missing values, or applying other data preprocessing techniques.

The model_trainer.py file can contain the necessary code for training your machine learning models. This might involve tasks such as data splitting, feature selection, model initialization, hyperparameter tuning, training loops, and model evaluation. Keeping the model training logic in a separate file helps in maintaining a clear separation of concerns and makes it easier to modify or enhance the training process without affecting other parts of the codebase.

in  the src i have crated a file named as logger.py, utils.py and exception.py

logger.py: This file can contain code related to logging, such as setting up loggers, defining logging levels, formatting log messages, and writing log entries to files or other outputs. It helps you manage and track the execution and behavior of your code.

utils.py: The utils.py file typically holds utility functions that can be used across different components of your project. These functions often perform common tasks or provide helpful functionality that can be reused throughout your codebase. It's a good place to keep functions that don't fit into a specific component but are still valuable for your project.

exception.py: The exception.py file can define custom exception classes or handle specific exception scenarios. By centralizing exception handling in one place, you can have consistent error handling across your codebase and improve the readability and maintainability of your project.

These additional files further enhance the organization and structure of your project, making it easier to navigate and maintain. It's a good practice to keep related code grouped together and separate concerns into different files.
also i have created a folder called pipeline that containesa the two of the following files prediction_pipeline.py and train_pipeline.py

prediction_pipeline.py: This file can contain code for the prediction pipeline, which includes loading trained models, preprocessing input data, and generating predictions. It could also involve post-processing steps or handling specific prediction-related tasks.

train_pipeline.py: The train_pipeline.py file can include code for the training pipeline, which encompasses tasks such as data ingestion, data transformation, model training, model evaluation, and model serialization. It represents the end-to-end process of training your machine learning models.

After execution of the  pip installl -r requirements.txt   we have  and egg .info file created showing the pacage we have created thus with this we can continue to delve deeper into our project.


## BELOW IS THE LOGGER.PY FILE

'''
import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,


)
'''
Import statements: The code begins by importing the necessary modules. It imports logging to use the Python logging module and os to work with file paths and directories. It also imports datetime to generate a timestamp for the log file.

Constants and variable setup: The code defines a constant named LOG_FILE that contains the filename for the log file. It uses the current timestamp generated by datetime.now() and formats it as "mm_dd_YYYY_HH_MM_SS.log".

Log file path creation: The code creates the log file path by joining the current working directory (os.getcwd()) with a subdirectory named "logs" and the LOG_FILE constant. It uses os.path.join() to ensure cross-platform compatibility.

Directory creation: The code uses os.makedirs() to create the directory specified by logs_path (the log file directory). The exist_ok=True argument ensures that the directory is created only if it doesn't already exist.

Log file path setup: The code creates another constant named LOG_FILE_PATH by joining logs_path with the LOG_FILE constant. This represents the full file path of the log file.

Basic logging configuration: The code uses logging.basicConfig() to configure the basic logging settings. It sets the following parameters:

filename: Specifies the file path of the log file (LOG_FILE_PATH).
format: Sets the log message format, including the timestamp (%(asctime)s), line number (%(lineno)d), logger name (%(name)s), log level (%(levelname)s), and the log message (%(message)s).
level: Sets the logging level to logging.INFO, which means only messages with an INFO level or higher (e.g., INFO, WARNING, ERROR) will be logged.
By executing this code in your logger.py file, you are creating a log file with a timestamped filename in the "logs" directory relative to your project's root directory. The log file will contain log messages with the specified format and logging level.

You can use the logging module throughout your project to log various events, messages, or errors. For example, you can use logging.info() to log informational messages or logging.error() to log error messages. These log entries will be written to the log file you configured in logger.py.

## HERE IS HOW THE EXCEPTION.PY  FILE LOOKS LIKE

'''
import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message

    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

'''
Import statements: The code imports sys module to work with system-related functionality and logging from the src.logger module, which represents the logger you defined in your logger.py file.

Function error_message_detail: This function takes an error object and an error_detail object (from sys.exc_info()) as input. It retrieves the traceback information from error_detail, including the filename and line number where the error occurred. It then constructs an error message string using this information and the original error message. The formatted error message is returned.

Class CustomException: This class inherits from the Exception class. It overrides the __init__() method to accept an error_message and an error_detail object. The error_message is passed to the parent Exception class via super().__init__() to set the error message. Additionally, the error_message_detail() function is called to construct an error message string with detailed information about the error, which is stored as self.error_message.

Method __str__(): This method overrides the default __str__() method of the Exception class. It returns the self.error_message, which represents the detailed error message.

By having this code in your exception.py file, you are defining a custom exception class (CustomException) that extends the functionality of the built-in Exception class. This allows you to raise and handle custom exceptions in your code.

The error_message_detail() function provides a way to generate detailed error messages by capturing traceback information and combining it with the original error message. This can be useful for logging and displaying informative error messages when exceptions occur.

Overall, this code helps you create custom exceptions with detailed error messages and integrates them with your logging system defined in logger.py.

we can test whether our exeption and logger files are working and we can basically basically add our  this lines to our logger .y  file
'''
if __name__=="__main__":
    logging.info("logging has started")
'''
 this will create a new file known as a logs file where it will contain loges of  files with the vrious steos and processes  that have been don to our code    the message should just be LOGGING HAS STARTED.  After this we will have to delete the little snippet  line of code above and then save the file.


 It is expected that a notebook has to be done for the model and the prediction we will have to import   the  complete notebook then from then we will start our pipelines . we will import the notebook and it should run using this environmen (remember we craeted python   3.9)

 ## ONSET OF OUR INGESTION.PY DATA FILE DOCUMENT
 '''
 import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

'''
HERE IS A brief  explanation of what the code s doing above
Import statements: The code imports necessary modules and classes from different sources, including the os module, sys module, custom exception class CustomException from exception.py, logging module from logger.py, pandas library for data handling, and train_test_split function from sklearn.model_selection for splitting the data.

DataIngestionConfig class: This class is a data class defined using the @dataclass decorator. It holds the configuration for data ingestion, including paths to the train, test, and raw data files. The paths are defined using os.path.join() to handle file path construction in a cross-platform manner. If the paths are not provided, default values are used.

DataIngestion class: This is the main class that performs data ingestion. It has an __init__() method that initializes an instance of the DataIngestionConfig class.

initiate_data_ingestion() method: This method is responsible for performing the data ingestion process. It begins by logging a message to indicate that the data ingestion process has started.

Data loading: The code reads a CSV file named 'stud.csv' using the pd.read_csv() function, which returns a pandas DataFrame. The code logs a message to indicate that the dataset has been read successfully.

Directory creation: The code creates directories for the train, test, and raw data paths using os.makedirs() to ensure the directories exist. It uses the exist_ok=True argument to avoid raising an error if the directories already exist.

Data saving: The code saves the entire dataset as the raw data file using the df.to_csv() function. It saves the train and test sets separately by splitting the dataset using the train_test_split() function. The train and test sets are then saved as CSV files using to_csv().

Logging: The code logs messages at various stages of the data ingestion process to provide information about the progress and status of the operation.

CustomException handling: If an exception occurs during the data ingestion process, it is caught using a try-except block. The exception is then raised as a CustomException with the original exception object (e) and sys module information.

Main execution: The code creates an instance of the DataIngestion class and calls the initiate_data_ingestion() method to start the data ingestion process.

In simple terms, this code reads a CSV file, splits it into train and test sets, and saves them to separate files. It also logs messages at each step to provide information and handles any exceptions that may occur during the process. The code follows a modular approach by using separate classes for configuration, exception handling, and logging.

When you check the logger.py file all the loggins.info statements must be present if they are present then the data has been entered into the system and we can move on to data transformation .
 wev will have to enter our artifacts to our gitignore file


 ## WE HAVE THE  DATA TRANSFORMATION.PY WHERE WE WILL BE PREPROCESSING OUR DATA
 '''
 import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

'''


Import statements: The code imports necessary modules and classes from different sources, including the sys module, dataclass decorator, numpy library for numerical operations, pandas library for data handling, various classes from the sklearn library for data preprocessing, custom exception class CustomException from exception.py, logging module from logger.py, and os module for handling file paths.

DataTransformationConfig class: This class is a data class defined using the @dataclass decorator. It holds the configuration for data transformation, including the file path for saving the preprocessor object. The file path is defined using os.path.join() to handle file path construction in a cross-platform manner. If the file path is not provided, a default value is used.

DataTransformation class: This is the main class that performs data transformation. It has an __init__() method that initializes an instance of the DataTransformationConfig class.

get_data_transformer_object() method: This method is responsible for creating a data transformer object that will be used for data preprocessing. It defines the numerical and categorical columns in the dataset and creates separate pipelines for each type of column. The numerical pipeline applies a median imputer and standard scaler, while the categorical pipeline applies a most frequent imputer, one-hot encoding, and standard scaler. The pipelines are then combined using the ColumnTransformer.

initiate_data_transformation() method: This method performs the data transformation process. It takes the paths to the train and test data files as input. The code reads the train and test data into pandas DataFrames. It then obtains the preprocessor object by calling the get_data_transformer_object() method.

Data preprocessing: The code separates the input features and the target feature from the train and test data. It applies the preprocessor object to the input features using the fit_transform() and transform() methods. The transformed input features and the target features are then combined into arrays.

Preprocessor object saving: The preprocessor object is saved using the save_object() function from the utils.py file. The file path and the preprocessor object are passed to the function for saving.

Logging: The code logs messages at various stages of the data transformation process to provide information about the progress and status of the operation.

CustomException handling: If an exception occurs during the data transformation process, it is caught using a try-except block. The exception is then raised as a CustomException with the original exception object (e) and sys module information.

In simple terms, this code defines a data transformation component that applies preprocessing techniques to the input data. It creates a preprocessor object using pipelines for numerical and categorical columns. The preprocessor object is used to transform the input features of the train and test datasets. The transformed features, along with the target features, are saved as arrays. The preprocessor object is also saved for future use. The code follows a modular approach by using separate classes for configuration, exception handling, and logging, as well as separate functions for saving objects and obtaining the preprocessor object.

## WE HAVE MODIFIED OUR DATA INGESTION.PY FILE ALSO

'''
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()


    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

'''

Import statements: You have added import statements for the DataTransformation class and DataTransformationConfig class from the src.components.data_transformation module.

train_data and test_data variables: After calling the initiate_data_ingestion() method, you are assigning the returned values (train data path and test data path) to the train_data and test_data variables.

DataTransformation instance: You create an instance of the DataTransformation class using data_transformation = DataTransformation().

Data transformation: You call the initiate_data_transformation() method of the DataTransformation instance, passing the train_data and test_data variables as arguments. The returned values (train array, test array, and preprocessor file path) are assigned to the train_arr, test_arr, and _ variables respectively.

In simple terms, the modified code performs data ingestion and data transformation operations. After data ingestion, it calls the initiate_data_transformation() method from the DataTransformation class to perform data transformation on the ingested data. The transformed data is stored in train_arr and test_arr variables for further use.


## WE HAVE THE MODEL TRAINER .PY FILE

'''
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)

'''
## HERE IS A BRIEF EXPLANATION OF WHAT THE CODE DOES


The ModelTrainer class in the code is responsible for training and evaluating different regression models on the given training and testing data. Here's a brief explanation of the code:

Import statements: The necessary libraries and classes are imported, including various regression models, evaluation metrics, and utility functions.

ModelTrainerConfig dataclass: It defines the configuration for the model trainer, including the file path to save the trained model.

initiate_model_trainer method: This method takes train_array and test_array as inputs, which are the transformed training and testing data obtained from the data transformation process.

Data splitting: The training and testing data arrays are split into input features (X_train, X_test) and target variables (y_train, y_test).

Model selection and hyperparameter tuning: Different regression models are defined along with their hyperparameter configurations in the models and params dictionaries. These models are evaluated using the evaluate_models function, which performs cross-validation and returns a dictionary of model scores.

Finding the best model: The model with the highest score is selected as the best model for further analysis. If the best model score is below 0.6, a custom exception is raised.

Saving the best model: The best model is saved as a serialized object using the save_object function, and the file path is defined in the ModelTrainerConfig.

Model evaluation: The best model is used to make predictions on the testing data, and the R-squared score is calculated to evaluate the model's performance.

Return value: The R-squared score is returned as the output of the initiate_model_trainer method.

In simple terms, the modified code trains multiple regression models on the transformed data and selects the best-performing model based on the R-squared score. The best model is then saved for future use.

## I HAVE ALSO MODIFIED MY DATA_INGESTION.PY FILE TO 

'''
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()


    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

'''

##HERE IS A BRIEF EXPLANATION OF WHAT THE CODE DOES
Additional import statements: The ModelTrainerConfig and ModelTrainer classes from the model_trainer.py module are imported.

ModelTrainerConfig import in the DataIngestionConfig dataclass: The ModelTrainerConfig class is imported and included in the dataclass decorator.

ModelTrainer initialization: An instance of the ModelTrainer class is created by instantiating the ModelTrainer object.

Model training and evaluation: After the data transformation step, the initiate_model_trainer method of the ModelTrainer class is called, passing the train_arr and test_arr as arguments. The resulting R-squared score is printed using print(modeltrainer.initiate_model_trainer(train_arr,test_arr)).

In summary, the modified code now performs both data ingestion and model training in sequence. It first ingests the data, performs the necessary transformations, and then trains and evaluates the models using the transformed data. Finally, the R-squared score of the best model is printed as the output.
