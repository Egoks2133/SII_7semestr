import pickle
import numpy as np
import tensorflow as tf

MODEL_PATH = "lstm_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

GENERATED_LENGTH = 200
TOP_K = 3

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

char_to_index = tokenizer["char_to_index"]
index_to_char = tokenizer["index_to_char"]
SEQUENCE_LENGTH = tokenizer["sequence_length"]


def generate_text(seed_text):

    if not seed_text:
        raise ValueError("Начальный текст не может быть пустым.")

    unknown_chars = [
        char for char in seed_text
        if char not in char_to_index
    ]

    if unknown_chars:
        raise ValueError(
            f"Неизвестные символы: {set(unknown_chars)}"
        )

    encoded = [
        char_to_index[char]
        for char in seed_text
    ]

    generated_text = seed_text

    for _ in range(GENERATED_LENGTH):

        input_sequence = encoded[-SEQUENCE_LENGTH:]

        if len(input_sequence) < SEQUENCE_LENGTH:
            padding = (
                [input_sequence[0]]
                * (SEQUENCE_LENGTH - len(input_sequence))
            )

            input_sequence = padding + input_sequence

        input_data = np.array(
            [input_sequence],
            dtype=np.int32
        )

        predictions = model.predict(
            input_data,
            verbose=0
        )[0]

        probabilities = tf.nn.softmax(
            predictions
        ).numpy()

        top_k = min(TOP_K, len(probabilities))

        top_indices = np.argsort(probabilities)[-top_k:]

        top_probabilities = probabilities[top_indices]

        top_probabilities = (
            top_probabilities / np.sum(top_probabilities)
        )

        next_index = np.random.choice(
            top_indices,
            p=top_probabilities
        )

        next_char = index_to_char[int(next_index)]

        generated_text += next_char
        encoded.append(int(next_index))

    return generated_text

seed = input("Введите начальный текст: ")

result = generate_text(seed)

print("\nСгенерированный текст:\n")
print(result)

print(
    f"\nСгенерировано символов: {GENERATED_LENGTH}"
)
