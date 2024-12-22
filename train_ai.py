import os
import tensorflow as tf
from official.nlp import optimization  # to create AdamW optimizer
from sentence_transformers import SentenceTransformer
import numpy as np
tf.get_logger().setLevel('ERROR')

with open("questions.txt", "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f]
with open("answers.txt", "r", encoding="utf-8") as f:
        answers = [line.strip() for line in f]

model_name = "DeepPavlov/rubert-base-cased"

embedding_model = SentenceTransformer(model_name)

question_embeddings = embedding_model.encode(questions)
answer_embeddings = embedding_model.encode(answers)

X = np.array([np.concatenate([q_emb, a_emb]) for q_emb in question_embeddings for a_emb in answer_embeddings])
Y = np.array([1 if i == j else 0 for i in range(len(questions)) for j in range(len(answers))])

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(1536,)))
model.add(tf.keras.layers.Dense(200, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
es=tf.keras.callbacks.EarlyStopping(monitor="auc", mode="max", patience=8, restore_best_weights=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(curve="pr", name="auc")])
model.fit(X,Y, epochs=1500, class_weight={0:1, 1:50}, callbacks=[es], batch_size=32)
model.summary()
model.save("chatbot.keras")
