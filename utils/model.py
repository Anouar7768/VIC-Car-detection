from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm
from feature_extractor import get_hog_features_function


def validate_model(model, X, y):
    """
    Test a model, and compare it to other models
    :param model: model to test
    :param X: car and non car features of images
    :param y: labels of each frame (1:car, 0:not a car)
    :return: accuracy, balanced_accuracy, f1_score
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)  # We could do a cross validation, but it will take a lot of time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_pred, y_test)
    print(f"accuracy is {accuracy}")
    print(f"balanced accuracy is {balanced_accuracy}")
    print(f"f1-score is {f1}")
    return accuracy, balanced_accuracy, f1


def train_model(model, X, y):
    """
    Train
    :param model: model to train on the whole dataset
    :param X: car and non car features of images
    :param y: labels of each frame (1:car, 0:not a car)
    :return: model
    """
    model.fit(X, y)  # We train this time on the whole dataset
    return model


def predict(model, img, window_size, sliding_step, confidence_threshold):
    """
    Predict and
    :param model: model already trained
    :param img: image where we want to detect cars
    :param window_size: the window size of the sliding window
    :param sliding_step: step at which we slide the window on the  image
    :param confidence_threshold: confidence threshold at which we consider a prediction
    :return:
    """
    h, w = img.shape[0], img.shape[1]
    pred = []
    for x in tqdm(range(0, h - window_size, sliding_step)):
        for y in range(0, w - window_size, sliding_step):
            sliding_img = img[x:x + window_size, y:y + window_size, :]
            fd = get_hog_features_function(sliding_img, 8, 16, 1, False)
            score = model.decision_function([fd])[0]
            y_pred = model.predict([fd])
            if y_pred == 1 and score >= confidence_threshold:
                pred.append([x, y, x + window_size, y + window_size])
    return pred
