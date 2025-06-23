from sentence_transformers import SentenceTransformer
import numpy as np
import time

class TextEmbedder:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Инициализация модели для преобразования текст <-> эмбеддинг
        :param model_name: название модели Sentence Transformers
        """
        self.model = SentenceTransformer(model_name)
        # Для простоты будем считать, что размерность эмбеддинга фиксирована
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def text_to_embedding(self, text: str) -> np.ndarray:
        """
        Преобразует текст в векторное представление (эмбеддинг)
        :param text: входной текст
        :return: numpy массив с эмбеддингом (размерность зависит от модели)
        """
        return self.model.encode(text, convert_to_tensor=False)


# Пример использования
if __name__ == "__main__":
    start = time.time()
    embedder = TextEmbedder()
    print(f'timer: {time.time() - start}')

    # Текст -> эмбеддинг
    text = "Пример текста для тестирования"
    start = time.time()
    embedding = embedder.text_to_embedding(text)
    print(f'timer: {time.time() - start}')
    print(f"Эмбеддинг: {embedding[:5]}...")