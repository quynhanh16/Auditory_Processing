import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import joblib

# TODO: Create a training and testing set and then do the k-fold cross validation.
# TODO: Calculate the R^2 before and after of the testing and training set.
# TODO: Change x and y of predicted vs. actual scatter plot have the same scale.
# TODO: The coefficient matrix should be 45 by 201
# TODO: Non-linear model.
# TODO: Try Ridge Model
# Frequency y axis (small to big).


def linear_model(X: np.ndarray, y: np.ndarray):
    """
    Fit a simple linear regression model with k-fold cross-validation using a training/testing split.

    Reports R² before and after training, plots predictions and coefficient heatmap.

    :param X: Feature matrix of shape (n_samples, n_features), where features = 201 * 45
    :param y: Target vector of shape (n_samples,)
    :return: None
    """

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    grid = GridSearchCV(pipeline, param_grid={}, cv=5, verbose=3,
                        scoring='neg_mean_squared_error', return_train_score=True)

    # Fit on training data only
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    joblib.dump(model, 'linear_model.pkl')

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Final R²
    print(f"\nFinal R² (train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"Final R² (test):  {r2_score(y_test, y_pred_test):.4f}")
    print(f"Final MSE (test): {mean_squared_error(y_test, y_pred_test):.4f}")

    # Plot predictions (test)
    plt.figure(figsize=(12, 4))
    plt.plot(y_test[:2000], color="black", label="Actual")
    plt.plot(y_pred_test[:2000], color="red", label="Prediction", linewidth=0.7)
    plt.legend()
    plt.xlim(left=0)
    plt.title("First 2000 Predictions (Test Set)")
    plt.tight_layout()
    plt.show()

    # Scatter plot: match scale
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred_test, y_test, color="black", s=1, alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # y = x line
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Predicted vs Actual")
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    plt.show()

    # Coefficient heatmap
    coefs = model.named_steps['model'].coef_

    print("Coefficient shape:", coefs.shape)
    try:
        coef_matrix = coefs.reshape(45, 201)  # reshape to (frequencies, time)
    except ValueError:
        print("Error reshaping coefficients. Expected shape (9045,) for reshape to (45, 201).")
        return

    sns.heatmap(np.abs(coef_matrix), cmap="coolwarm", yticklabels=[f"f{i}" for i in range(45)],
                cbar_kws={'label': 'Coefficient Magnitude'})
    plt.title("Coefficient Heatmap (freq × time)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
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
    joblib.dump(model, 'linear_model_lasso.pkl')
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
    plt.xlim(left=0)
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