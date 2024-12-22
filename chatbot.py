from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf
import os

with open("answers.txt", "r", encoding="utf-8") as f:
    answers = [line.strip() for line in f]

model_name = "DeepPavlov/rubert-base-cased"

embedding_model = SentenceTransformer(model_name)

answer_embeddings = embedding_model.encode(answers)

chatbot = tf.keras.models.load_model("chatbot.h5")

print("Добро пожаловать, я могу ответить на ваши вопросы вопросы из жизни университета/факультета/кафедры. Напишите свой вопрос:")
while True:
    question=input().strip()
    if question.lower() == 'exit':
        print("До скорой встречи.")
        break
    q_emb = embedding_model.encode([question])[0]
    probabilities=[chatbot.predict(np.concatenate([q_emb, a_emb]).reshape(1, -1), verbose=0)[0][0] for a_emb in answer_embeddings]
    print("Ответ:", answers[np.argmax(probabilities)])

    print("У вас ещё есть вопросы? Для завершения работы введите 'exit'. Ваш вопрос:")
