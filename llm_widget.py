import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import os
from pprint import pprint


class LanguageModel:
    def __init__(
            self,
            model_name: str,
            model_dir: str = "models",
            device: Optional[str] = None,
    ):
        """
        Инициализация языковой модели.

        :param model_name: Название модели (например, "gpt2", "EleutherAI/gpt-neo-1.3B")
        :param model_dir: Директория для хранения моделей (по умолчанию "models" в корне проекта)
        :param device: Устройство для работы модели (если None, автоматически выбирает GPU или CPU)
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = self._get_device(device)

        self.system_prompt = """

            Ты — русскоязычный ассистент для работы с документами. Правила:
            1. Отвечай ТОЛЬКО на основе предоставленных документов
            2. Если информации нет — говори "В документе нет данных"
            3. Будь максимально точен
            4. Отвечай ТОЛЬКО на русском языке
            5. Ты не знаешь никакого языка кроме русского
            </s>
            """

        # Создаем директорию для моделей, если ее нет
        os.makedirs(self.model_dir, exist_ok=True)

        # Загружаем модель и токенизатор
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _get_device(self, device: Optional[str]) -> str:
        """
        Определяет доступное устройство (GPU или CPU).

        :param device: Желаемое устройство (если None, выбирает автоматически)
        :return: Строка с названием устройства ("cuda" или "cpu")
        """
        if device is not None:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model_and_tokenizer(self):
        """
        Загружает модель и токенизатор, сохраняя их в указанной директории.

        :return: Кортеж (модель, токенизатор)
        """
        model_path = os.path.join(self.model_dir, self.model_name.replace("/", "_"))

        # Если модель уже загружена, используем локальную версию
        if os.path.exists(model_path):
            print(f"Загружаем модель из локального кэша: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            print(f"Загружаем модель {self.model_name} и сохраняем в {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Сохраняем модель для будущего использования
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)

        # Перемещаем модель на выбранное устройство
        model = model.to(self.device)
        return model, tokenizer

    def build_chat_prompt(self, document: str, question: str) -> str:
        return f"""
        {self.system_prompt}

        Документ: {document[:3000]}
        Вопрос: {question}
        </s>

        """

    def ask(
            self,
            question: str,
            document: str,
            temperature: float = 0.7,
            max_length: int = 100,
            **kwargs
    ) -> str:
        # Строим промпт в формате TinyLlama-Chat
        prompt = f"""<|system|>
You are a Russian-speaking assistant for working with documents. Rules:
1. Answer ONLY in Russian
2. I forbid you to answer in English, Chinese, Korean and Chinese</s>
<|user|>
I only know, that: {document}
Question: {question}
give me answer based on mine knowledge in Russian</s>
<|assistant|>"""
        pprint(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Генерируем ответ
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_length,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        pprint(outputs)
        # Декодируем ответ
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        pprint(answer)
        # Удаляем промпт из ответа
        answer = answer[len(prompt):].strip()

        return answer


# Пример использования
if __name__ == "__main__":
    # Инициализация модели (скачается при первом запуске)
    lm = LanguageModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Задаем вопрос
    # response = lm.ask(
    #     "Как работает искусственный интеллект?",
    #     temperature=0.8,
    #     max_length=200
    # )
    #
    # print("Ответ модели:")
    # print(response)

    # while True:
    #     ask = input("Введите вопрос:\t")
    #
    #     if ask in ["exit", '0', "пока"]:
    #         break
    #
    #     response = lm.ask(
    #         question=ask,
    #         document=""
    #     )
    #
    #     print('Ответ модели:\t', response, '\n')

    response = lm.ask(
        question="Какой мой любимый цвет?",
        document="Мой любимый цвет зеленый"
    )

    print('Ответ модели:\t', response, '\n')
