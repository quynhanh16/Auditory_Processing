import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import least_squares
import joblib


# Nonlinear functions
def sigmoid(params, res):
    a, b, c = params
    return a / (1 + np.exp(b * (c - res)))


def fl(params, res):
    a, b, c, s = params
    return b + a * np.exp(-np.exp(c * (res - s)))


def hyperbolic_tan(params, res):
    a, b, c = params
    result = a * np.tanh(b * (res - c))
    return result * (result > 0)


# Residual functions
def residuals_sigmoid(params, x, r):
    return r - sigmoid(params, x)


def residuals_fl(params, x, r):
    return r - fl(params, x)


def residuals_tanh(params, x, r):
    return r - hyperbolic_tan(params, x)


# Main regression pipeline
def nonlinear_pipeline(X: np.ndarray, y: np.ndarray):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    nonlinear_models = {
        'sigmoid': (sigmoid, residuals_sigmoid, [np.max(y_train), 0.1, 0.1]),
        'fl': (fl, residuals_fl, [1, 1, 0.1, 0.1]),
        'tanh': (hyperbolic_tan, residuals_tanh, [np.max(y_train), 0.1, 0.1])
    }

    best_score = -np.inf
    best_nl_model = None
    best_predictions = None
    best_nl_name = None
    best_params = None

    for name, (nl_func, residual_func, theta) in nonlinear_models.items():
        print(f"\n--- Testing Nonlinear Function: {name} ---")
        all_preds, all_true = [], []

        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Linear pipeline
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
            pipe.fit(X_fold_train, y_fold_train)

            # Linear predictions
            lin_preds = pipe.predict(X_fold_val)
            mean_r = np.mean(y_fold_val, axis=0)

            # Nonlinear fitting
            try:
                nl_model = least_squares(residual_func, theta, args=(lin_preds, mean_r))
                nonlin_preds = nl_func(nl_model.x, lin_preds)
            except Exception as e:
                print(f"Error fitting {name}: {e}")
                continue

            all_preds.append(nonlin_preds)
            all_true.append(y_fold_val)

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)

        score = r2_score(all_true, all_preds)
        print(f"R² for {name}: {score:.4f}")

        # Save the best model for this nonlinear function
        # Final training on full training set for this function
        final_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        final_pipe.fit(X_train, y_train)
        joblib.dump(final_pipe, f'best_linear_pipeline_{name}.pkl')

        linear_preds_test = final_pipe.predict(X_test)
        nl_fit = least_squares(nonlinear_models[name][1], theta, args=(linear_preds_test, y_test))
        joblib.dump(nl_fit.x, f'best_{name}_params.pkl')

        if score > best_score:
            best_score = score
            best_nl_model = nl_func
            best_params = theta
            best_nl_name = name
            best_predictions = nl_func(nl_fit.x, linear_preds_test)

    print(f"\nBest Nonlinear Function: {best_nl_name} with R²: {best_score:.4f}")

    # Plotting results
    plt.figure(figsize=(12, 4))
    plt.plot(y_test[:2000], color="black", label="Actual")
    plt.plot(best_predictions[:2000], color="red", label="Prediction", linewidth=0.7)
    plt.legend()
    plt.title(f"Best Model ({best_nl_name}) - First 2000 Predictions")
    plt.tight_layout()
    plt.savefig("best_model_predictions.png")  # Save plot instead of showing
    plt.close()

    # Scatter plot
    min_val = min(y_test.min(), best_predictions.min())
    max_val = max(y_test.max(), best_predictions.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(best_predictions, y_test, color="black", s=1, alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Predicted vs Actual")
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png")  # Save plot instead of showing
    plt.close()

    # Coefficient heatmap
    coefs = final_pipe.named_steps['model'].coef_
    try:
        coef_matrix = coefs.reshape(45, 201)
        sns.heatmap(np.abs(coef_matrix), cmap="coolwarm",
                    yticklabels=[f"f{i}" for i in range(45)],
                    cbar_kws={'label': 'Coefficient Magnitude'})
        plt.title("Coefficient Heatmap (freq × time)")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("coefficient_heatmap.png")  # Save plot instead of showing
        plt.close()
    except Exception as e:
        print("Error displaying heatmap:", e)
