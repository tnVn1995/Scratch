import scipy.linalg as linalg
import numpy as np
from typing import List
from sklearn import datasets
from sklearn.linear_model import LinearRegression as LinearR
import matplotlib.pyplot as plt

plt.style.use('seaborn')
import time


class LinearRegression:
    """Perform a linear regression on the data"""

    def __init__(self, R_squared: float = 0.0, rmse: float = 0.0) -> None:
        """[Linear Regression model metrics]

        Args:
            R_squared (float, optional): [coefficient determination value]. Defaults to 0.0.
            rmse (float, optional): [root mean squared error]. Defaults to 0.0.
        """
        self.R_squared = R_squared
        self.rmse = rmse

    def fit(self, X: List[float], y: List[float], norm: bool = True):
        """[Fit a regression line on the data]

        Args:
            X (List[float]): [data]
            y (List[float]): [target]
            norm (bool, optional): [whether to normalize]. Defaults to True.

        Returns:
            LR object
        """
        if norm:
            X = (X - np.mean(X)) / np.linalg.norm(X)
        ones = np.ones(X.shape[0])[:, np.newaxis]
        X = np.hstack((ones, X))
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = X.dot(beta)
        assert np.array_equal(y_hat.shape, y.shape)

        numerator = np.sum(np.square(y - y_hat))
        y_mean = np.mean(y)
        denominator = np.sum(np.square(y - y_mean))

        self.R_squared = 1 - numerator / denominator
        self.rmse = np.sqrt(np.mean(numerator))
        self.coef_ = beta[1:]
        self.intercept_ = beta[0]
        self.preds = y_hat
        return self

    def RMSE(self, y_true, preds):
        """Return the rmse of the fitted model"""
        numerator = np.sum(np.square(y_true - preds))

        rmse = np.sqrt(np.mean(numerator))
        return rmse


def main():
    print('[INFO] Fitting scratched LR on the boston dataset...')
    X, y = datasets.load_boston(return_X_y=True)
    x = X[:, 1].reshape(-1, 1)
    LR = LinearRegression()
    LR.fit(x, y)
    time.sleep(1)

    # plt.plot(x,, color = 'k')

    print('[INFO] Fitting sklearn LR on the boston dataset ...')
    lr = LinearR()
    lr.fit(x, y)
    preds = lr.predict(x)
    time.sleep(1)
    print('[INFO] Compare model performance ...')
    time.sleep(1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(x, y, color='b')
    ax1.plot(x, LR.preds, color='r')
    ax1.set_title('Scratched LR fitted line')
    ax2.scatter(x, y, color='b')
    ax2.plot(x, preds, color='orange')
    ax2.set_title('sklearn LR fitted line')
    plt.show()


# import plotly.graph_objects as go
# fig = go.Figure()

# fig.add_trace(go.Scatter(x=x.squeeze(), y=y, mode='markers', name='Scatterplot of data'))
# fig.add_trace(go.Line(x=x.squeeze(), y=preds, name='Sklearn Fitted Line'))
# fig.add_trace(go.Line(x=x.squeeze(), y=LR.preds, name='Scratched LR Fitted Line'))
# fig.update_layout(title='Comparison of Sklearn and Scratched models')


if __name__ == "__main__":
    main()
