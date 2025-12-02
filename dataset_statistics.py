"""
Este script calcula las distribuciones de clases para los atributos de puntuación inicial, puntuación final y capitalización
en el dataset de entrenamiento, leyendo los datos desde archivos Parquet batches. Esto lo usamos para definir los pesos de la
función de pérdida en el loop de entrenamiento.

Para cada problema de clasificación, el peso por clase es 1-f, siendo f la frecuencia con la que aparece en el dataset de
**entrenamiento**.
"""
from collections import Counter
import pandas as pd
import numpy as np
import os

from TP2.dataloader import punct_inicial_to_idx, punct_final_to_idx


def compute_class_weights(base_path: str, parquet_file_subpath: str, batch_size: int, instance_count: int):
    total_punct_ini_counts = Counter()
    total_punct_fin_counts = Counter()
    total_cap_counts = Counter()

    num_batches = (instance_count + batch_size - 1) // batch_size

    total_instances = 0
    for batch_number in range(num_batches):
        file_name = f"{parquet_file_subpath}{batch_number}.parquet"
        file_path = os.path.join(base_path, file_name)
        df = pd.read_parquet(file_path)

        total_instances += len(df)
        punct_ini_counts = Counter(df['punt_inicial'].map(punct_inicial_to_idx))
        punct_fin_counts = Counter(df['punt_final'].map(punct_final_to_idx))
        cap_counts = Counter(df['capitalización'])

        total_punct_ini_counts.update(punct_ini_counts)
        total_punct_fin_counts.update(punct_fin_counts)
        total_cap_counts.update(cap_counts)

    def compute_weights(counter: Counter):
        class_weights = {cls: count / total_instances for cls, count in counter.items()}
        return class_weights

    punct_ini_weights = compute_weights(total_punct_ini_counts)
    punct_fin_weights = compute_weights(total_punct_fin_counts)
    cap_weights = compute_weights(total_cap_counts)

    return punct_ini_weights, punct_fin_weights, cap_weights

if __name__ == "__main__":
    base_path = "/home/jorge/tmp"
    parquet_file_subpath = "train_tokens_batch_"
    batch_size = 16384
    instance_count = 180625  # Cambiar por la cantidad real de instancias en el dataset

    punct_ini_weights, punct_fin_weights, cap_weights = compute_class_weights(
        base_path, parquet_file_subpath, batch_size, instance_count
    )

    print("Punctuation Initial Weights:", punct_ini_weights)
    print("Punctuation Final Weights:", punct_fin_weights)
    print("Capitalization Weights:", cap_weights)