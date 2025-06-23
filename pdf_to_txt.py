import pdfplumber
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def pdf_to_text(input_pdf: str | Path, output_txt: str | Path) -> None:
    """Конвертирует PDF в текстовый файл.

    Args:
        input_pdf: Путь к исходному PDF (относительно PROJECT_ROOT или абсолютный)
        output_txt: Путь для сохранения TXT (относительно PROJECT_ROOT или абсолютный)
    """
    # Приводим пути к абсолютным
    input_path = PROJECT_ROOT / input_pdf if not Path(input_pdf).is_absolute() else Path(input_pdf)
    output_path = PROJECT_ROOT / output_txt if not Path(output_txt).is_absolute() else Path(output_txt)

    # Создаем директорию для выходного файла, если её нет
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(input_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"  # Добавляем пустую строку между страницами

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    # Пример использования с новой структурой
    pdf_to_text(
        input_pdf="storage/pdf/tk.pdf",
        output_txt="storage/processed/txt/tk_rf.txt"
    )