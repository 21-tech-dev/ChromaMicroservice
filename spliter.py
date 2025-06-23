import re
import json
from pathlib import Path

# Получаем абсолютный путь к корню проекта (app/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def split_articles_to_json(
        input_txt: str | Path,
        output_json: str | Path,
        pattern: str = r"(?<=\n)\s*Статья\s(\d+)\."
) -> None:
    """Разбивает текст ТК РФ на статьи и сохраняет в JSON.

    Args:
        input_txt: Путь к текстовому файлу
        output_json: Путь для сохранения JSON
        pattern: Регулярное выражение для разделения статей
    """
    # Приводим пути к абсолютным
    input_path = PROJECT_ROOT / input_txt if not Path(input_txt).is_absolute() else Path(input_txt)
    output_path = PROJECT_ROOT / output_json if not Path(output_json).is_absolute() else Path(output_json)

    # Чтение исходного файла
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Разделение на статьи с сохранением номеров
    articles = re.split(pattern, text)

    # Первый элемент - всё до Статьи 1 (игнорируем)
    articles_data = {}
    for i in range(1, len(articles), 2):
        if i + 1 < len(articles):
            article_num = articles[i].strip()
            article_text = articles[i + 1].strip()
            articles_data[f"Статья {article_num}"] = article_text

    # Сохранение JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    split_articles_to_json(
        input_txt="storage/processed/txt/tk_rf.txt",
        output_json="storage/processed/json/tk_rf.json"
    )