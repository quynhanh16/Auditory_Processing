import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


# TODO: Create a training and testing set and then do the k-fold cross validation.
# TODO: Calculate the R^2 before and after of the testing and training set.
# TODO: Change x and y of predicted vs. actual scatter plot have the same scale.
# TODO: The coefficient matrix should be 45 by 201
# TODO: Non-linear model.
# TODO: Try Ridge Model
# Frequency y axis (small to big).

def linear_model(X: np.ndarray, y: np.ndarray):
    """
    Fitting a simple linear regression model with k-folds as cross-validation.

    :param X:
    :param y:
    :return:
    """
    pipeline = Pipeline([
        # ('poly', PolynomialFeatures(include_bias=False)),
        # ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    param_grid = {
        # 'poly__degree': [1, 2],
        # 'model__alpha': [0.001, 0.1, 1.0, 10.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=3, scoring='neg_mean_squared_error', return_train_score=True)
    grid.fit(X, y)

    print("\nCross-Validation Results:")
    for mean_train, mean_test, params in zip(
            grid.cv_results_['mean_train_score'],
            grid.cv_results_['mean_test_score'],
            grid.cv_results_['params']
    ):
        print(f"{params}: Train MSE = {-mean_train:.4f}, Test MSE = {-mean_test:.4f}")

    model = grid.best_estimator_
    # degree = grid.best_params_['poly__degree']
    y_pred = model.predict(X)

    final_mse = mean_squared_error(y, y_pred)
    final_r2 = r2_score(y, y_pred)

    # print(f"\nSelected Polynomial Degree: {degree}")
    print(f"Final MSE: {final_mse:.4f}")
    print(f"Final R²: {final_r2:.4f}")

    plt.plot(y[:2000], color="black", label="Actual")
    plt.plot(y_pred[:2000], color="red", label="Prediction", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.scatter(y_pred, y, color="black", s=1, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Coefficient heatmap
    coefs = model.named_steps['model'].coef_
    print("Coefficient", coefs.shape)

    coef_df = np.abs(coefs.reshape(1, -1))  # reshape to 2D for heatmap
    sns.heatmap(coef_df, cmap="coolwarm", yticklabels=[], cbar_kws={'label': 'Coefficient Magnitude'})
    plt.title("Coefficient Heatmap")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def linear_model_standardized(X: np.ndarray, y: np.ndarray):
    """
    Fitting a simple linear regression model with k-folds as cross-validation.

    :param X:
    :param y:
    :return:
    """
    pipeline = Pipeline([
        # ('poly', PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    param_grid = {
        # 'poly__degree': [1, 2],
        # 'model__alpha': [0.001, 0.1, 1.0, 10.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=3, scoring='neg_mean_squared_error', return_train_score=True)
    grid.fit(X, y)

    print("\nCross-Validation Results:")
    for mean_train, mean_test, params in zip(
            grid.cv_results_['mean_train_score'],
            grid.cv_results_['mean_test_score'],
            grid.cv_results_['params']
    ):
        print(f"{params}: Train MSE = {-mean_train:.4f}, Test MSE = {-mean_test:.4f}")

    model = grid.best_estimator_
    # degree = grid.best_params_['poly__degree']
    y_pred = model.predict(X)

    final_mse = mean_squared_error(y, y_pred)
    final_r2 = r2_score(y, y_pred)

    # print(f"\nSelected Polynomial Degree: {degree}")
    print(f"Final MSE: {final_mse:.4f}")
    print(f"Final R²: {final_r2:.4f}")

    plt.plot(y[:2000], color="black", label="Actual")
    plt.plot(y_pred[:2000], color="red", label="Prediction", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.scatter(y_pred, y, color="black", s=1, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Coefficient heatmap
    coefs = model.named_steps['model'].coef_
    print("Coefficient", coefs.shape)

    coef_df = np.abs(coefs.reshape(1, -1))  # reshape to 2D for heatmap
    sns.heatmap(coef_df, cmap="coolwarm", yticklabels=[], cbar_kws={'label': 'Coefficient Magnitude'})
    plt.title("Coefficient Heatmap")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def linear_model_lasso(X: np.ndarray, y: np.ndarray):
    """
    Fitting a simple linear regression model with k-folds as cross-validation.

    :param X:
    :param y:
    :return:
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(max_iter=1000))
    ])

    param_grid = {
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        verbose=3,
        scoring='neg_mean_squared_error',
        return_train_score=True
    )

    grid.fit(X, y)

    print("\nCross-Validation Results:")
    for mean_train, mean_test, params in zip(
        grid.cv_results_['mean_train_score'],
        grid.cv_results_['mean_test_score'],
        grid.cv_results_['params']
    ):
        print(f"{params}: Train MSE = {-mean_train:.4f}, Test MSE = {-mean_test:.4f}")

    model = grid.best_estimator_
    best_alpha = grid.best_params_['model__alpha']
    y_pred = model.predict(X)

    final_mse = mean_squared_error(y, y_pred)
    final_r2 = r2_score(y, y_pred)

    print(f"\nBest Alpha: {best_alpha}")
    print(f"Final MSE: {final_mse:.4f}")
    print(f"Final R²: {final_r2:.4f}")

    # Time-series plot of predictions
    plt.plot(y[:2000], color="black", label="Actual")
    plt.plot(y_pred[:2000], color="red", label="Prediction", linewidth=0.7)
    plt.legend()
    plt.title("First 2000 Predictions")
    plt.tight_layout()
    plt.show()

    # Scatter plot
    plt.scatter(y_pred, y, color="black", s=1, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.show()

    # Coefficient heatmap
    coefs = model.named_steps['model'].coef_
    print("Coefficient shape:", coefs.shape)

    coef_df = np.abs(coefs.reshape(1, -1))
    sns.heatmap(coef_df, cmap="coolwarm", yticklabels=[], cbar_kws={'label': 'Coefficient Magnitude'})
    plt.title("Lasso Coefficient Heatmap")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()