import fire
import pickle
import logging

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """Loads model from `model_path`"""
    with open(model_path, 'rb') as file:
        saved_model = pickle.load(file)
        return saved_model

def save_model(model, model_path: str):
    """Saves `model` to `model_path`"""
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

class Classifier:
    """
    Some classifier, that makes some random classifications.
    """

    def train(self, train_data_path: str, model_path: str, k: int = 5):
        """
        Trains model on `train_data_path` data and saves trained model to `model_path`.
        Additionaly you can set KNN classifier `k` parameter.
        :param train_data_path: path to train data in csv format
        :param model_path: path to save model to.
        :param k: k-neighbors parameter of model.
        """
        logger.info(f"Loading train data from {train_data_path} ...")
        df = pd.read_csv(train_data_path)
        X = df.drop(columns=['y'])
        y = df['y']

        logger.info("Running model training...")
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)

        logger.info(f"Saving model to {model_path} ...")
        save_model(model, model_path)

        logger.info("Successfully trained model.")

    def predict(self, predict_data_path: str, model_path: str, output_path: str):
        """
        Predicts `predict_data_path` data using `model_path` model and saves predictions to `output_path`
        :param predict_data_path: path to data for predictions
        :param model_path: path to trained model
        :param output_path: path to save predictions
        """
        logger.info(f"Loading data for predictions from {predict_data_path} ...")
        X = pd.read_csv(predict_data_path)

        logger.info(f"Loading model from {model_path} ...")
        model = load_model(model_path)

        logger.info("Running model predictions...")
        y_pred = model.predict(X)

        logger.info(f"Saving predictions to {output_path} ...")
        pd.DataFrame(y_pred).to_csv(output_path)

        logger.info("Successfully predicted.")

if __name__ == "__main__":
    fire.Fire(Classifier)