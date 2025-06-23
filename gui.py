from pprint import pprint

from llm_widget import LanguageModel
import chromadb
from chromadb import ClientAPI
from embedding_worker import TextEmbedder

from fill_db_story import get_sentences_by_embedding
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Union
import time
import threading

load_dotenv()


class LLMWidget(tk.Frame):
    def __init__(self, master,
                 widget_name_label: str,
                 llm_model: Union[LanguageModel, str],
                 client: Union[ClientAPI, None]):
        super().__init__(master=master)
        self.lm = llm_model
        if self.lm == 'GigaChat':
            self.chat = GigaChat(
                credentials=os.getenv("GIGACHAT_AUTH_DATA_V"),
                scope="GIGACHAT_API_PERS",
                verify_ssl_certs=False,
                profanity_check=False,
                timeout=30
            )

        self.chromo_client = client
        self.embedder = TextEmbedder()
        self.widget_name_label = widget_name_label

        self.configure(bg='#f0f0f0', padx=10, pady=10)
        self.widget_configuration()

    def validate_numeric_input(self, P):
        """Валидация для ввода только чисел и точки"""
        if P == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    def widget_configuration(self):
        # Стили
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('TEntry', font=('Arial', 10), padding=5)
        style.configure('TCombobox', font=('Arial', 10), padding=5)
        style.configure('Disabled.TEntry', foreground='#888888')
        style.configure('Disabled.TCombobox', foreground='#888888')

        # Регистрация функции валидации
        vcmd = (self.register(self.validate_numeric_input), '%P')

        # Главный фрейм диалога
        self.dialog_frame = ttk.Frame(self)
        self.main_frame_label = ttk.Label(
            self.dialog_frame,
            text=self.widget_name_label,
            font=('Arial', 12, 'bold'),
            foreground='#333333'
        )

        self.dialog = scrolledtext.ScrolledText(
            self.dialog_frame,
            wrap=tk.WORD,
            height=15,
            font=('Arial', 10),
            bg='white',
            padx=10,
            pady=10,
            relief=tk.FLAT,
            highlightbackground='#cccccc',
            highlightthickness=1
        )
        self.dialog.config(state='disabled')

        self.main_frame_label.pack(side=tk.TOP, pady=(0, 10))
        self.dialog.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        self.dialog_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        # Фрейм вопроса
        self.question_frame = ttk.Frame(self)
        self.question_entry = ttk.Entry(
            self.question_frame,
            width=50
        )
        self.send_question_button = ttk.Button(
            self.question_frame,
            text="Отправить",
            command=self.on_send_question_button_clicked,
            style='TButton'
        )

        self.question_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))
        self.send_question_button.pack(side=tk.LEFT)
        self.question_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))

        self.instruction_frame = ttk.Frame(self, style='TFrame')

        # Выбор базы данных и время ответа
        self.db_time_frame = ttk.Frame(self.instruction_frame)

        # Выбор базы данных
        self.db_selector_frame = ttk.Frame(self.db_time_frame)
        self.db_selector_label = ttk.Label(
            self.db_selector_frame,
            text="Выбор базы данных:",
            font=('Arial', 10)
        )
        self.db_selector = ttk.Combobox(
            self.db_selector_frame,
            values=["ТК РФ", "Репка"],
            state="readonly",
            width=20
        )
        self.db_selector.current(0)

        self.db_selector_label.pack(side=tk.LEFT, padx=(0, 10))
        self.db_selector.pack(side=tk.LEFT)
        self.db_selector_frame.pack(side=tk.LEFT, padx=(0, 20))

        # Время ответа
        self.response_time_frame = ttk.Frame(self.db_time_frame)
        self.response_time_label = ttk.Label(
            self.response_time_frame,
            text="Время ответа:",
            font=('Arial', 10)
        )
        self.response_time_value = ttk.Label(
            self.response_time_frame,
            text="0.00 сек",
            font=('Arial', 10),
            foreground='#555555'
        )

        self.response_time_label.pack(side=tk.LEFT, padx=(0, 5))
        self.response_time_value.pack(side=tk.LEFT)
        self.response_time_frame.pack(side=tk.LEFT)

        self.db_time_frame.pack(fill=tk.X, pady=(0, 10))

        # Настройки температуры
        self.temperature_frame = ttk.Frame(self.instruction_frame)
        self.temperature_label_entry = ttk.Label(
            self.temperature_frame,
            text="Температура:",
            width=25,
            anchor='e'
        )
        self.temperature_entry = ttk.Entry(
            self.temperature_frame,
            width=10,
            validate='key',
            validatecommand=vcmd
        )
        self.temperature_entry.insert(0, "0.3")

        self.temperature_label_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.temperature_entry.pack(side=tk.LEFT)
        self.temperature_frame.pack(fill=tk.X, pady=(0, 10))

        # Настройки размера ответа
        self.answer_token_frame = ttk.Frame(self.instruction_frame)
        self.answer_token_size_label_entry = ttk.Label(
            self.answer_token_frame,
            text="Макс. размер ответа в токенах:",
            width=25,
            anchor='e'
        )
        self.answer_token_size_entry = ttk.Entry(
            self.answer_token_frame,
            width=10,
            validate='key',
            validatecommand=vcmd
        )
        self.answer_token_size_entry.insert(0, "512")

        self.answer_token_size_label_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.answer_token_size_entry.pack(side=tk.LEFT)
        self.answer_token_frame.pack(fill=tk.X)

        self.instruction_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))

        # Если модель GigaChat, делаем поля ввода недоступными
        if self.lm == "GigaChat":
            self.temperature_entry.config(state='disabled', style='Disabled.TEntry')
            self.answer_token_size_entry.config(state='disabled', style='Disabled.TEntry')

        # Настройка веса строк и столбцов
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

    def on_send_question_button_clicked(self):
        question = self.question_entry.get()
        if not question.strip():
            messagebox.showwarning("Предупреждение", "Введите вопрос перед отправкой")
            return

        doc_dict = {"ТК РФ": "tk_rf",
                    "Репка": "fairy_tale"}

        # Засекаем время начала
        start_time = time.time()

        # Получаем выбранную базу данных
        selected_db = self.db_selector.get()

        collection = self.chromo_client.get_collection(doc_dict[selected_db])

        try:
            result = get_sentences_by_embedding(
                embedding=self.embedder.text_to_embedding(question),
                collection=collection,
                n_results=3
            )
            pprint(['Результат векторного поиска: \n', result])

            if not self.lm == 'GigaChat':
                temperature = float(self.temperature_entry.get())
                self.response = self.lm.ask(
                    question=question,
                    document=result,
                    temperature=temperature
                )
            else:
                messages = [
                    SystemMessage(
                        content=f"Ты ассистент, который отвечает на вопросы пользователя, оперируя только следующей информацией\n{result}"),
                    HumanMessage(content=question)
                ]
                self.response = self.chat.invoke(input=messages).content

            self.dialog.config(state='normal')
            self.dialog.delete("1.0", tk.END)
            self.dialog.insert(tk.END, self.response)
            self.dialog.config(state='disabled')

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
        finally:
            # Обновляем время ответа
            response_time = time.time() - start_time
            self.response_time_value.config(text=f"{response_time:.2f} сек")


class LoadingScreen:
    def __init__(self, root: tk.Tk, title: str = "Загрузка"):
        self.root = root
        self.loading_window = tk.Toplevel(root)
        self.loading_window.title(title)
        self.loading_window.geometry("300x100")
        self.loading_window.resizable(False, False)

        # Центрирование окна
        self.loading_window.grab_set()  # Модальное окно
        self._center_window(self.loading_window)

        # Элементы интерфейса
        self.label = ttk.Label(
            self.loading_window,
            text="Инициализация приложения...",
            font=('Arial', 10)
        )
        self.label.pack(pady=10)

        self.progress = ttk.Progressbar(
            self.loading_window,
            orient="horizontal",
            length=200,
            mode="indeterminate"
        )
        self.progress.pack(pady=10)
        self.progress.start(10)

        # Скрываем основное окно пока идет загрузка
        root.withdraw()

    def _center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'+{x}+{y}')

    def close(self):
        self.progress.stop()
        self.loading_window.destroy()
        self.root.deiconify()  # Показываем основное окно


class LLMApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LLM Chat Interface")

        # Показываем окно загрузки
        self.loading_screen = LoadingScreen(self.root)

        # Запускаем загрузку в отдельном потоке
        self.loading_thread = threading.Thread(
            target=self._initialize_resources,
            daemon=True
        )
        self.loading_thread.start()

        # Проверяем завершение загрузки
        self._check_loading_complete()

        self.root.mainloop()

    def _initialize_resources(self):
        """Инициализация ресурсоемких объектов"""
        try:
            # Инициализация клиента ChromaDB
            self.chroma_client = chromadb.HttpClient(host='localhost', port=8000)

            # Инициализация LanguageModel (может быть долгой)
            self.lm = LanguageModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

            self.initialization_success = True
        except Exception as e:
            self.initialization_error = str(e)
            self.initialization_success = False

    def _check_loading_complete(self):
        """Проверяем завершение загрузки и создаем интерфейс"""
        if self.loading_thread.is_alive():
            # Если загрузка еще идет, проверяем снова через 100 мс
            self.root.after(2000, self._check_loading_complete)
        else:
            # Закрываем окно загрузки
            self.loading_screen.close()

            if not self.initialization_success:
                # Показываем ошибку, если что-то пошло не так
                tk.messagebox.showerror(
                    "Ошибка инициализации",
                    f"Не удалось инициализировать приложение:\n{self.initialization_error}"
                )
                self.root.destroy()
                return

            # Создаем основной интерфейс
            self._create_interface()

    def _create_interface(self):
        """Создание основного интерфейса приложения"""
        # Создаем фреймы для моделей
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Первая модель (TinyLlama)
        frst_llm = LLMWidget(
            main_frame,
            'TinyLlama',
            self.lm,
            client=self.chroma_client
        )
        frst_llm.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Вторая модель (GigaChat)
        second_llm = LLMWidget(
            main_frame,
            'GigaChat',
            'GigaChat',
            client=self.chroma_client
        )
        second_llm.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)


if __name__ == "__main__":
    app = LLMApplication()

# if __name__ == "__main__":
#     root = tk.Tk()
#     lm = LanguageModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#
#     chroma_client = chromadb.HttpClient(host='localhost', port=8000)
#     frst_llm = LLMWidget(
#         root,
#         'Первая модель',
#         lm,
#         client=chroma_client)
#     second_llm = LLMWidget(root,
#                            'GigaChat',
#                            'GigaChat',
#                            client=chroma_client)
#
#     frst_llm.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
#     second_llm.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
#
#     root.mainloop()
