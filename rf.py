from TP2.tokenizer import dataset_to_tokens
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_and_process_dataset(dataset_path):
    dataset = load_dataset("csv", data_files=dataset_path)
    labels = dataset['train']['Y']
    df_full = dataset_to_tokens(labels)
    return df_full

def aggregate_word_embeddings(df_tokens):
    # para usar el RF por palabra
    df_word = df_tokens.groupby(['instancia_id', 'numero_palabra']).agg({
        'embedding': lambda x: np.mean(np.stack(x), axis=0),
        'punt_inicial': 'first',
        'punt_final': 'first',
        'capitalización': 'first',
        'token': lambda x: ''.join([t.replace('##','') for t in x])
    }).reset_index().rename(columns={'token':'palabra'})
    return df_word


def expand_embeddings(df_word):
    # expandir y agregar feature de pos en palabra
    emb = np.vstack(df_word['embedding'])
    emb_df = pd.DataFrame(emb, index=df_word.index)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]
    df_word = pd.concat([df_word, emb_df], axis=1)

    
    df_word['posicion_palabra'] = df_word['numero_palabra']
    df_word['dist_final'] = (
        df_word.groupby('instancia_id')['numero_palabra'].transform('max')
        - df_word['numero_palabra']
    )
    feature_cols = [f"emb_{i}" for i in range(emb_df.shape[1])] + ['posicion_palabra'] + ['dist_final']
    
    return df_word, feature_cols


def train_rf_task_instance_split(df_word, feature_cols, task, test_size=0.35):
    instancia_ids = df_word['instancia_id'].unique()
    train_ids, test_ids = train_test_split(instancia_ids, test_size=test_size, random_state=42)

    train_mask = df_word['instancia_id'].isin(train_ids)
    test_mask  = df_word['instancia_id'].isin(test_ids)

    X_train = df_word.loc[train_mask, feature_cols]
    y_train = df_word.loc[train_mask, task]
    X_test  = df_word.loc[test_mask, feature_cols]
    y_test  = df_word.loc[test_mask, task]

    
    rf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
  

    y_pred_test = rf.predict(X_test)
    df_word.loc[test_mask, f"{task}_pred"] = y_pred_test


    print(classification_report(y_test, y_pred_test))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, cmap='Blues', values_format='d')
    plt.title(f'Matriz de Confusión: {task}')
    plt.show()
    return df_word, rf



if __name__ == "__main__":
    # Dataset para entrenar RF
    dataset_path = "TP2/dataset/raw_clean_data/arg_speech_dataset.csv"
    df_tokens = load_and_process_dataset(dataset_path)
    print(df_tokens)
    df_word = aggregate_word_embeddings(df_tokens)
    df_word, feature_cols = expand_embeddings(df_word)
    print(df_word)
    tasks = ['punt_inicial', 'punt_final', 'capitalización']
    for task in tasks:
        df_word, _ = train_rf_task_instance_split(df_word, feature_cols, task)

    
