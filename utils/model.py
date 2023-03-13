from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score



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



