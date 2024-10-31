import tensorflow as tf
from sklearn.metrics import mean_squared_error

from tools.utils import load_state

model = tf.keras.models.load_model('1Dx2-CNN.keras')

X = load_state('x_model.pkl')
y = load_state('resp_m18_t(27, 889.5).pkl')

x = model.predict(X)
mse = mean_squared_error(y, x)

print("Mean Squared Error:", mse)
