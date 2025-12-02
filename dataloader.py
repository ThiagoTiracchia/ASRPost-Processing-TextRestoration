import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset

# mapeo
punct_inicial_to_idx = {
    "": 0,
    "¿": 1,
}

idx_to_punct_inicial = {v: k for k, v in punct_inicial_to_idx.items()}

punct_final_to_idx = {
    "": 0,
    "','": 1,
    ".": 2,
    "?": 3,
}

idx_to_punct_final = {v: k for k, v in punct_final_to_idx.items()}

class TextDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        parquet_file_subpath: str,
        batch_size: int,
        instance_count: int,
        shuffle_batches: bool = False,
        random_seed: int = 1683,
        max_seq_len: int | None = None,
    ):
        self._base_path = base_path
        self._parquet_file_subpath = parquet_file_subpath
        self._batch_size = batch_size
        self._instance_count = instance_count
        self._loaded_batch_idx = -1
        self._max_seq_len = max_seq_len

        np.random.seed(random_seed)
        self._instance_id_shuffle = None
        self._shuffle_batches = shuffle_batches


    def __len__(self):
        return self._instance_count

    def _load_parquet_batch_if_needed(self, idx: int):
        batch_number = idx // self._batch_size
        if batch_number == self._loaded_batch_idx:
            return
        file_name = f"{self._parquet_file_subpath}{batch_number}.parquet"
        file_path = os.path.join(self._base_path, file_name)
        df = pd.read_parquet(file_path)
        df["punt_inicial"] = df["punt_inicial"].map(punct_inicial_to_idx)
        df["punt_final"] = df["punt_final"].map(punct_final_to_idx)
        self.instances = df.set_index(["instancia_id", "numero_palabra"])

        last_instance_id = df["instancia_id"].iloc[-1]
        if self._shuffle_batches:
            self._instance_id_shuffle = np.random.permutation(last_instance_id + 1)
        else:
            self._instance_id_shuffle = np.arange(last_instance_id + 1)

        self._loaded_batch_idx = batch_number

    def __getitem__(self, idx):
        """
        Como tenemos más datos de los que nos entran en memoria, los separamos en batches (no son los mismos batches que
        usa el DataLoader de pytorch). Cada batch está guardado en un archivo parquet distinto.
        Al pedir un ítem, cargamos el batch correspondiente si es que no está cargado ya.
        Luego, devolvemos el ítem correspondiente dentro del batch cargado.

        Es importante no hacer shuffle de los idx ya que estaríamos recargando constantemente los batches. Se podría,
        si uno quiere, hacer shuffle dentro de cada batch en _load_parquet_batch_if_needed.
        """
        self._load_parquet_batch_if_needed(idx)
        effective_idx = idx % self._batch_size
        effective_idx = self._instance_id_shuffle[effective_idx] # mapear al idx efectivo si es que estamos haciendo shuffle (dentro de este batch, no entra todo el dataset en memoria)
        emb = torch.tensor(np.vstack(self.instances.loc[effective_idx]["embedding"]), dtype=torch.float32)
        punt_ini = torch.tensor([self.instances.loc[effective_idx]["punt_inicial"].to_list()], dtype=torch.long)
        punt_fin = torch.tensor([self.instances.loc[effective_idx]["punt_final"].to_list()], dtype=torch.long)
        caps = torch.tensor([self.instances.loc[effective_idx]["capitalización"].to_list()], dtype=torch.long)

        if self._max_seq_len is not None and emb.size(0) > self._max_seq_len:
            truncate_to = self._max_seq_len
            emb = emb[:truncate_to]
            punt_ini = punt_ini[:, :truncate_to]
            punt_fin = punt_fin[:, :truncate_to]
            caps = caps[:, :truncate_to]
        return emb, punt_ini, punt_fin, caps

    def collate_fn(self, batch):
        embeddings = [item[0] for item in batch]
        punt_inicial = [item[1] for item in batch]
        punt_final = [item[2] for item in batch]
        capitalizacion = [item[3] for item in batch]

        return embeddings, punt_inicial, punt_final, capitalizacion