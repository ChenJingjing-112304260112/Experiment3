import tensorflow as tf
import numpy as np
import pandas as pd

print("=== Testing Model ===")

# Load model
model = tf.keras.models.load_model('model.h5')
print("[OK] Model loaded successfully")

# Load test data
test_df = pd.read_csv('test.csv')
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Test first 10 samples
predictions = model.predict(X_test[:10], verbose=0)
print("\nPredictions for first 10 samples:")
for i, pred in enumerate(predictions):
    digit = np.argmax(pred)
    confidence = pred[digit] * 100
    print(f"Sample {i+1}: Digit={digit}, Confidence={confidence:.2f}%")

print("\n[OK] Model test completed!")