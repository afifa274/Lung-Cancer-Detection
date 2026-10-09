import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Load & preprocess ────────────────────────────────────────────────────────
df = pd.read_csv('cancer patient data sets.csv')

# Map text labels once; skip if already numeric
if df['Level'].dtype == object:
    level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Level'] = df['Level'].map(level_mapping)
    df.to_csv('cancer patient data sets.csv', index=False)

# Features: columns 2 → second-to-last; target: last column
X = df.iloc[:, 2:-1].values
y = df.iloc[:, -1].values.astype(int)

print(f"Dataset shape: {X.shape}  |  Classes: {np.bincount(y)}")

# ── Scale ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / test split (stratified) ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=SEED, stratify=y
)

y_train_cat = keras.utils.to_categorical(y_train, 3)
y_test_cat  = keras.utils.to_categorical(y_test,  3)

# ── Model ────────────────────────────────────────────────────────────────────
def build_model(input_dim: int = 23) -> Sequential:
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax'),   # softmax is correct for multiclass
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',  # correct loss for one-hot labels
        metrics=['accuracy'],
    )
    return model

model = build_model(X_train.shape[1])
model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
]

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train_cat,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    verbose=2,
)

# ── Evaluate ─────────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f'\nTest accuracy: {round(accuracy * 100, 2)} %')

y_pred = np.argmax(model.predict(X_test), axis=1)
print('\nClassification report:')
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))

# ── Save training plots ───────────────────────────────────────────────────────
os.makedirs('static/images', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric, title in zip(axes, ['accuracy', 'loss'], ['Model Accuracy', 'Model Loss']):
    ax.plot(history.history[metric],     label='Train')
    ax.plot(history.history[f'val_{metric}'], label='Validation')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('static/images/training_history.png', dpi=120)
plt.close()
print('Training plots saved to static/images/training_history.png')

# ── Persist artefacts ─────────────────────────────────────────────────────────
model.save('cancer_snus_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print('Model saved → cancer_snus_model.h5')
print('Scaler saved → scaler.pkl')
