import pickle
import numpy as np
import tensorflow as tf

DATASET_PATH = "dataset.txt"
MODEL_PATH = "model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

SEQUENCE_LENGTH = 80
EMBEDDING_DIM = 128
LSTM_UNITS = 256
BATCH_SIZE = 64
EPOCHS = 30

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    text = file.read()

print(f"Загружено символов: {len(text)}")

vocabulary = sorted(set(text))

char_to_index = {
    char: index for index, char in enumerate(vocabulary)
}

index_to_char = {
    index: char for char, index in char_to_index.items()
}

vocab_size = len(vocabulary)

print(f"Размер словаря: {vocab_size}")

encoded_text = np.array(
    [char_to_index[char] for char in text],
    dtype=np.int32
)

sequences = np.lib.stride_tricks.sliding_window_view(
    encoded_text,
    SEQUENCE_LENGTH + 1
)

X = sequences[:, :-1]
y = sequences[:, -1]

print(f"Количество обучающих последовательностей: {len(X)}")

dataset = tf.data.Dataset.from_tensor_slices((X, y))

dataset = dataset.shuffle(
    min(len(X), 10000)
).batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,)),

    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM
    ),

    tf.keras.layers.LSTM(
        LSTM_UNITS,
        return_sequences=False
    ),

    tf.keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["accuracy"]
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    shuffle=False
)

model.save(MODEL_PATH)

with open(TOKENIZER_PATH, "wb") as file:
    pickle.dump({
        "char_to_index": char_to_index,
        "index_to_char": index_to_char,
        "sequence_length": SEQUENCE_LENGTH
    }, file)

print("Обучение завершено!")
print(f"Модель сохранена: {MODEL_PATH}")
print(f"Словарь сохранен: {TOKENIZER_PATH}")
