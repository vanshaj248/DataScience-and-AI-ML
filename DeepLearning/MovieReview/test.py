import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── 1. Load Dataset ───────────────────────────────────────
# Option A: via keras (auto-download)
(X_train_raw, y_train), (X_test_raw, y_test) = \
    keras.datasets.imdb.load_data(num_words=10000)

# Decode integer sequences back to text
word_index = keras.datasets.imdb.get_word_index()
inv_index  = {v+3: k for k, v in word_index.items()}
inv_index.update({0:'', 1:'', 2:'', 3:''})
decode = lambda seq: ' '.join(inv_index.get(i, '?') for i in seq)

train_texts = [decode(s) for s in X_train_raw]
test_texts  = [decode(s) for s in X_test_raw]
print("Sample review:", train_texts[0][:200])
print("Label:", y_train[0])  # 1=positive, 0=negative

# ── 2. Class Distribution ─────────────────────────────────
vals, counts = np.unique(y_train, return_counts=True)
plt.bar(['Negative','Positive'], counts, color=['salmon','steelblue'])
plt.title('Class Distribution (Train)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

# ─────────────────────────────────────────────────────────
# MODEL A: TF-IDF + Logistic Regression (Classical ML)
# ─────────────────────────────────────────────────────────

# ── 3. TF-IDF Vectorization ───────────────────────────────
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tr_tfidf = tfidf.fit_transform(train_texts)
X_te_tfidf = tfidf.transform(test_texts)

# ── 4. Train Logistic Regression ─────────────────────────
lr = LogisticRegression(max_iter=300, random_state=42)
lr.fit(X_tr_tfidf, y_train)
y_pred_lr = lr.predict(X_te_tfidf)

print("=== Logistic Regression Results ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr)*100:.2f}%")
print(classification_report(y_test, y_pred_lr,
      target_names=['Negative','Positive']))

# ROC Curve
y_prob_lr = lr.predict_proba(X_te_tfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
auc = roc_auc_score(y_test, y_prob_lr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'LR (AUC={auc:.3f})', color='royalblue')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Logistic Regression')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve_lr.png')
plt.show()

# ─────────────────────────────────────────────────────────
# MODEL B: LSTM Neural Network (Deep Learning)
# ─────────────────────────────────────────────────────────

# ── 5. Pad Sequences ──────────────────────────────────────
MAX_LEN = 200
X_tr_pad = pad_sequences(X_train_raw, maxlen=MAX_LEN, padding='post')
X_te_pad = pad_sequences(X_test_raw,  maxlen=MAX_LEN, padding='post')

# ── 6. Build LSTM Model ───────────────────────────────────
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64,
                     input_length=MAX_LEN),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
], name='LSTM_Sentiment')

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ── 7. Train LSTM ─────────────────────────────────────────
history = model.fit(
    X_tr_pad, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ── 8. Evaluate ───────────────────────────────────────────
_, lstm_acc = model.evaluate(X_te_pad, y_test, verbose=0)
print(f"\nLSTM Test Accuracy: {lstm_acc*100:.2f}%")

# ── 9. Training Curves ────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('LSTM Accuracy'); ax1.legend()
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('LSTM Loss'); ax2.legend()
plt.tight_layout()
plt.savefig('lstm_curves.png')
plt.show()

# ── 10. Confusion Matrix (LSTM) ───────────────────────────
y_pred_lstm = (model.predict(X_te_pad) > 0.5).astype(int).flatten()
cm = confusion_matrix(y_test, y_pred_lstm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
plt.title('Confusion Matrix – LSTM')
plt.tight_layout()
plt.savefig('confusion_lstm.png')
plt.show()

# ── 11. Predict Custom Review ─────────────────────────────
def predict_sentiment(review_text):
    seq = [[word_index.get(w.lower(), 2)+3 for w in review_text.split()]]
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    score = model.predict(padded)[0][0]
    label = "POSITIVE" if score > 0.5 else "NEGATIVE"
    print(f"Review   : {review_text[:80]}...")
    print(f"Sentiment: {label} (confidence: {score:.3f})")

predict_sentiment("This movie was absolutely brilliant and deeply moving!")
predict_sentiment("Terrible film, waste of time and completely boring.")