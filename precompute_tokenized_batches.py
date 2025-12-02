import os.path
import pandas as pd
from tokenizer import dataset_to_tokens
import os
from multiprocessing import Pool


RANDOM_SEED = 42
N_CPUS = 3
TOKENIZER_BATCH_SIZE = 8320

dataset_path = "TP2\\dataset\guardar_datos"


def process_single_batch(args):
    """Worker function that processes one batch."""
    batch_data, batch_idx, split_name, dataset_path = args
    tokenized = dataset_to_tokens(batch_data)
    output_path = os.path.join(dataset_path, f"{split_name}_tokens_batch_{batch_idx}.parquet")
    tokenized.to_parquet(output_path)
    return batch_idx


def batch_tokenizer():
    train_df = pd.read_parquet(dataset_path + "\train_dataset.parquet")
    train_labels = train_df['X']
    print(f"Train instances: {len(train_labels)}")
    train_tasks = [
        (train_labels[i:i + TOKENIZER_BATCH_SIZE], i // TOKENIZER_BATCH_SIZE, 'train', dataset_path)
        for i in range(0, len(train_labels), TOKENIZER_BATCH_SIZE)
    ]

    valid_df = pd.read_parquet(dataset_path + "\valid_dataset.parquet")
    valid_labels = valid_df['X']
    print(f"Validation instances: {len(valid_labels)}")
    valid_tasks = [
        (valid_labels[i:i + TOKENIZER_BATCH_SIZE], i // TOKENIZER_BATCH_SIZE, 'valid', dataset_path)
        for i in range(0, len(valid_labels), TOKENIZER_BATCH_SIZE)
    ]

    all_tasks = train_tasks + valid_tasks
    # Cada proceso usa MUCHA memoria, cuidado con esta opción (dejen htop abierto o un kill -9 listo y cuando se empiece
    # a trabar el cursor maten el proceso y prueben con menos cpus)
    with Pool(processes=N_CPUS) as pool:
        results = pool.map(process_single_batch, all_tasks)

    print(f"Completed {len(results)} batches using {N_CPUS} cores")

# yo no lo descargo porque ya lo tengo
#batch_tokenizer()
#exit()

if __name__ == "__main__":
    print(f"Este script va a escribir ~22GB de datos en {dataset_path}")
    while True:
        print("Continuar? (y/n)")
        answer = input().strip().lower()
        if answer == 'y':
            break
        elif answer == 'n':
            exit()
    batch_tokenizer()