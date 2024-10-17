import os
import time

import pandas as pd
from dotenv import load_dotenv

from app.models.yandexgpt import YandexGPT
from app.utils.submit import generate_submit


if __name__ == "__main__":
    load_dotenv()

    system_prompt = """
    Как цифровой помощник, твоя задача — помогать студенту, как это сделал бы преподаватель, не давая полных решений, а лишь предоставляя подсказки, наведения для самостоятельного размышления. Вот что тебе нужно сделать:

    НИ В КОЕМ СЛУЧАЕ НЕЛЬЗЯ ОТПРАВЛЯТЬ РЕШЕНИЕ 

1. Проанализируй код, который предоставил студент, но не объясняй всё детально. Намекни на место, где есть ошибка.
2. Если есть синтаксическая ошибка, намекни, в каком участке кода она произошла, но вместо готового исправления предложи студенту подумать, как это можно поправить. Например: «Внимательно посмотри на синтаксис этой строки. Кажется, здесь не хватает чего-то важного для правильного выполнения команды».
3. Если ошибка логическая, предложи подумать над тем, как результат программы меняется в зависимости от логики.
4. Убедись, что тон твоего ответа поддерживающий и наставнический, чтобы студент чувствовал, что он на правильном пути и может найти решение самостоятельно.

Важно: студент будет тебе отправлять тебе запросы, а ты должен давать конретные советы. Решение говорить НЕ надо. Только подсказки
    """

    yandex_gpt = YandexGPT(
        token=os.environ["YANDEX_GPT_IAM_TOKEN"],
        folder_id=os.environ["YANDEX_GPT_FOLDER_ID"],
        system_prompt=system_prompt,
    )


    def predict(row: pd.Series) -> str:
        if (row.name == 0):
            yandex_gpt.send_promt()
            time.sleep(6)
        # if (row.name >= 4):
        #     return "some"
        tmp = yandex_gpt.ask(row["student_solution"])
        time.sleep(0.5)
        return tmp


    generate_submit(
        test_solutions_path="data/raw/test/solutions.xlsx",
        predict_func=predict,
        save_path="data/processed/submission_3.csv",
        use_tqdm=True,
    )
