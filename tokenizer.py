import re
from enum import Enum
from functools import cache
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import BertTokenizer, BertModel

from models.simple_rnn import SimpleRNN
from dataloader import idx_to_punct_inicial, idx_to_punct_final


class Capitalizaciones(Enum):
    ALL_LOWERCASE = 0
    INITIAL_UPPERCASE = 1
    MIDDLE_UPPERCASE = 2
    ALL_UPPERCASE = 3

    @classmethod
    def from_int(cls, i: int) -> 'Capitalizaciones':
        try:
            target = int(i)
        except Exception:
            raise ValueError(f"Invalid integer value for Capitalizaciones: {i!r}")
        for member in cls:
            if member.value == target:
                return member
        raise ValueError(f"No Capitalizaciones member with value {target}")

    @classmethod
    def from_string(cls, string: str) -> 'Capitalizaciones':
        string = string.replace('.', '').replace(',', '').replace('?', '').replace('¿', '')
        if string.islower():
            return cls.ALL_LOWERCASE
        elif string[0].isupper() and string[1:].islower():
            return cls.INITIAL_UPPERCASE
        elif string.isupper():
            return cls.ALL_UPPERCASE
        else:
            return cls.MIDDLE_UPPERCASE

    def capitalizar(self, string: str) -> str:
        match self:
            case Capitalizaciones.ALL_LOWERCASE:
                return string.lower()
            case Capitalizaciones.INITIAL_UPPERCASE:
                return string.capitalize()
            case Capitalizaciones.MIDDLE_UPPERCASE:
                return string #TODO: hacer estrategia maestra acá...
            case Capitalizaciones.ALL_UPPERCASE:
                return string.upper()

@cache
def get_tokenizer_and_bert_embeddings():
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    embeddings_matrix = model.get_input_embeddings().weight.detach().numpy()
    return tokenizer, embeddings_matrix

def dataset_to_tokens(corpus_list: Iterable) -> pd.DataFrame:
    tokenizer, embeddigs_matrix = get_tokenizer_and_bert_embeddings()
    df_rows = []
    instance_id = -1

    for instance in corpus_list:
        sentence_words = instance.split(' ')
        sentence_words = [word.strip() for word in sentence_words if word.strip() != '']
        sentence_words = list(filter(lambda w: len(w) > 0, sentence_words))
        if sentence_words:
            instance_id += 1 # este código es pésimo :(
        try:
            for word_number, word in enumerate(sentence_words):
                assert word != ""
                normalized_word = word.lower().replace('.', '').replace(',', '').replace('?', '').replace('¿', '')

                capitalizacion = Capitalizaciones.from_string(word)
                tokens = tokenizer.tokenize(normalized_word)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                embedding = np.mean(embeddigs_matrix[token_ids], axis=0)

                classification_for_token = [["", "", f"{capitalizacion.value}"].copy() for _ in range(len(tokens))]
                if word[0] == "¿":
                    classification_for_token[0][0] = "¿"
                if word[-1] in ("?", ".", ","):
                    classification_for_token[-1][1] = word[-1] if word[-1] != "," else "','" # TODO: ver si son comillas simples o dobles
                for token_id, token, token_classes in zip(token_ids, tokens, classification_for_token):
                    df_rows.append([instance_id, token_id, token] + token_classes + [embedding, word_number])
        except Exception as e:
            print(f"ERROR EN EL STRING: {sentence_words}")
            print(f"INSTANCIA OG: {instance}")
            continue

    df = pd.DataFrame(df_rows, columns=["instancia_id","token_id", "token","punt_inicial","punt_final","capitalización", "embedding", "numero_palabra"])
    df["capitalización"] = df["capitalización"].astype(int)
    return df


def reconstruir_token_a_normalizada(df: pd.DataFrame) -> List[str]:
    oraciones = []
    for id_instancia, instancia_df in df.groupby(df.columns[0], sort=False):
        oracion = ""
        for token in instancia_df['token']:
            palabra_a_apendear = token
            if palabra_a_apendear.startswith("##"):
                palabra_a_apendear = palabra_a_apendear[2:]
                oracion += palabra_a_apendear
            else:
                 oracion += " " + palabra_a_apendear
        oraciones.append(oracion)
    oraciones_finales = []
    for oracion1 in oraciones:
        oracion1 = oracion1[1:]  # quitar espacio inicial
        # quiero un re.sub que reemplace "palabra ' s" por "palabra's"
        oracion1 = re.sub(r"\s' (\w+)", r"'\1", oracion1)
        oraciones_finales.append(oracion1)
    return oraciones_finales


def sentence_from_model_output(token_dataframe: pd.DataFrame, tokenizer: BertTokenizer) -> pd.DataFrame:
    missing_columns = {"instancia_id", "token_id", "token", "punt_inicial", "punt_final", "capitalización", "numero_palabra"} - set(token_dataframe.columns)
    assert not missing_columns, f"Missing columns: {missing_columns}"
    assert token_dataframe.columns[0] == "instancia_id"
    assert token_dataframe.columns[-1] == "numero_palabra"

    output_df_rows = []
    for id_instancia, instancia_df in token_dataframe.groupby(token_dataframe.columns[0], sort=False): # instancia_id column
        palabras_instancia = []
        for numero_palabra, instancia_una_palabra_df in instancia_df.groupby(instancia_df.columns[-1], sort=False): # numero_palabra column
            palabra = tokenizer.convert_tokens_to_string(list(instancia_una_palabra_df['token']))
            capitalizacion = Capitalizaciones.from_int(instancia_una_palabra_df['capitalización'].iloc[0])
            palabra_capitalizada = capitalizacion.capitalizar(palabra)
            puntuacion_final = instancia_una_palabra_df['punt_final'].iloc[-1]
            puntuacion_final = puntuacion_final[1] if puntuacion_final == "','" else puntuacion_final
            palabra_puntuada = instancia_una_palabra_df['punt_inicial'].iloc[0] + palabra_capitalizada + puntuacion_final
            palabras_instancia.append(palabra_puntuada)
        output_df_rows.append([id_instancia, " ".join(palabras_instancia)])
    return pd.DataFrame(output_df_rows, columns=["id_instancia", "instancia"])

class PunctuationPredictor:
    def __init__(self, model):
        self._model = model
        self._model.eval()

    def predict(self, corpus: str) -> str:
        sentence_words = self._corpus_to_word_list(corpus)
        embedding_tensor = self._corpus_to_embedding_tensor(sentence_words).to("cuda" if torch.cuda.is_available() else "cpu")
        out_punct_ini, out_punct_fin, out_cap = self._model(embedding_tensor)
        out_punct_ini, out_punct_fin, out_cap = out_punct_ini.squeeze(0), out_punct_fin.squeeze(0), out_cap.squeeze(0)
        _, pred_punct_ini = torch.max(out_punct_ini, dim=1)
        _, pred_punct_fin = torch.max(out_punct_fin, dim=1)
        _, pred_cap = torch.max(out_cap, dim=1)

        punct_inicial = [idx_to_punct_inicial[idx.item()] for idx in pred_punct_ini]
        punct_final = [idx_to_punct_final[idx.item()] for idx in pred_punct_fin]
        capitalizaciones = [Capitalizaciones.from_int(idx.item()) for idx in pred_cap]
        return self._punctuate_word_list(sentence_words, punct_inicial, punct_final, capitalizaciones)

    def _punctuate_word_list(self, word_list: List[str], punct_inicial: List[str], punct_final: List[str], capitalizaciones: List[Capitalizaciones]) -> str:
        punctuated_words = []
        for word, p_ini, p_fin, cap in zip(word_list, punct_inicial, punct_final, capitalizaciones):
            capitalized_word = cap.capitalizar(word)
            punctuated_word = f"{p_ini}{capitalized_word}{p_fin if p_fin != "','" else ','}"
            punctuated_words.append(punctuated_word)
        return ' '.join(punctuated_words)

    def _corpus_to_embedding_tensor(self, sentence_words: List[str]) -> Tensor:
        tokenizer, embeddigs_matrix = get_tokenizer_and_bert_embeddings()
        embeddings = []
        for word_number, word in enumerate(sentence_words):
            assert word != ""
            # No debería estar en la entrada pero por las dudas...
            normalized_word = word.lower().replace('.', '').replace(',', '').replace('?', '').replace('¿', '')
            tokens = tokenizer.tokenize(normalized_word)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            embedding = np.mean(embeddigs_matrix[token_ids], axis=0)
            embeddings.append(embedding)

        embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
        return embedding_tensor

    def _corpus_to_word_list(self, corpus: str) -> list[str]:
        sentence_words = corpus.split(' ')
        sentence_words = [word.strip() for word in sentence_words if word.strip() != '']
        sentence_words = list(filter(lambda w: len(w) > 0, sentence_words))
        return sentence_words


if __name__ == "__main__":
    model = SimpleRNN(embedding_dim=768, hidden_dim=384, num_layers=2, bidirectional=True)
    model.load_state_dict(torch.load("/home/jorge/oldhome/Documents/UBA/Carrera/2C2025/AprendizajeAutomaticoArgentino/aprendizaje_automatico/TP2/model_epoch_44.pth", map_location=torch.device('cpu')))
    model.to(torch.device('cpu'))
    pp = PunctuationPredictor(model)
    for s in ["hola como estas", "¿como te llamas", "este es un ejemplo de oracion", "me gusta programar en python", "la inteligencia artificial es el futuro", "necesito comprar manzanas peras uvas y naranjas"]:
        print(f"Input: {s}")
        print(f"Output: {pp.predict(s)}")
        print()

    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    dataset = load_dataset("csv", data_files="/home/jorge/oldhome/Documents/UBA/Carrera/2C2025/AprendizajeAutomaticoArgentino/aprendizaje_automatico/TP2/dataset/valid_dataset.csv")
    words = dataset['train']['X']
    labels = dataset['train']['Y']

    df = dataset_to_tokens(labels, tokenizer)
    output_df = sentence_from_model_output(df, tokenizer)
    """
    pass

