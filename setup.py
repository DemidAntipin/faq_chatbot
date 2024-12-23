import os
import subprocess

def install_requirements():
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

def download_model(model_name, local_path):
    print(f"Загрузка модели {model_name}...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(model_name)
    embedding_model.save(local_path)
    print(f"Модель сохранена в {local_path}")

if __name__ == "__main__":
    # Установка зависимостей
    install_requirements()

    # Загрузка модели
    model_name = "DeepPavlov/rubert-base-cased"
    local_model_path = "rubert_model"
    download_model(model_name, local_model_path)

    print("Установка завершена.")
