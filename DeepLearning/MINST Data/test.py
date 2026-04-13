import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# ── 1. Load MNIST Dataset ─────────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 2. Visualize Sample Images ────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.suptitle('Sample MNIST Images')
plt.tight_layout()
plt.savefig('mnist_samples.png')
plt.show()

# ── 3. Preprocessing ──────────────────────────────────────
# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# Reshape for CNN input: (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_test_oh  = keras.utils.to_categorical(y_test,  10)

# ── 4. Build CNN Model ────────────────────────────────────
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
], name='CNN_MNIST')

model.summary()

# ── 5. Compile Model ──────────────────────────────────────
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── 6. Train Model ────────────────────────────────────────
history = model.fit(
    X_train, y_train_oh,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ── 7. Evaluate on Test Set ───────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ── 8. Plot Training History ──────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch')
ax1.legend()
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch')
ax2.legend()
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# ── 9. Confusion Matrix ───────────────────────────────────
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ── 10. Classification Report ─────────────────────────────
print(classification_report(y_test, y_pred,
      target_names=[str(i) for i in range(10)]))

# ── 11. Visualize Predictions ─────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(10,4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28,28), cmap='gray')
    ax.set_title(f'Pred:{y_pred[i]} True:{y_test[i]}',
                 color='green' if y_pred[i]==y_test[i] else 'red',
                 fontsize=9)
    ax.axis('off')
plt.suptitle('Sample Predictions (green=correct, red=wrong)')
plt.tight_layout()
plt.savefig('predictions.png')
plt.show()

# ── 12. Save Model ────────────────────────────────────────
model.save('mnist_cnn_model.h5')
print("Model saved as mnist_cnn_model.h5")