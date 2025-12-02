from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from models.simple_gru import SimpleGru
from models.fine_tuningBert import BertPunctuationHead, build_finetuning_model
from dataloader import TextDataset
from training import training_loop

RANDOM_SEED = 42
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
"""
Breve explicación de nuestra clase Dataset. Cuando tokenizamos todo nustro dataset, nos dimos cuenta que el dataframe no
entraba en memoria. Por lo tanto, decidimos guardar el dataset tokenizado en varios archivos parquet, cada uno con un
batch de instancias (este batch no es lo mismo que el batch de entrenamiento, es qué partición del dataset tenemos en memoria
en un momento dado).

Al recibir el pedido de la instancia i, la clase TextDataset calcula a qué archivo parquet corresponde esa instancia, carga ese archivo
en memoria si es que no está ya cargado, y devuelve la instancia i desde ese archivo.

Es importante no hacer shuffle en el DataLoader, porque si no el hit-rate de la caché que tenemos en memoria sería de prácticamente 0
y estaríamos intercambiando archivos del disco en casi todas las instancias. Como una especie de parche sobre esto podemos hacer shuffle
dentro de uno de los batches que tenemos en memoria usando el parámetro shuffle_batches del constructor.
"""
BASE_PATH = "dataset\guardar_datos"
train_dataset = TextDataset(
    base_path=BASE_PATH,
    parquet_file_subpath="train_tokens_batch_",
    shuffle_batches=True,
    batch_size=5320,
    instance_count=20000,
    max_seq_len=512,
)
train_dataloader = DataLoader(train_dataset, batch_size=285, collate_fn=train_dataset.collate_fn) # NO HACER SHUFFLE!

valid_dataset = TextDataset(
    base_path=BASE_PATH,
    parquet_file_subpath="valid_tokens_batch_",
    batch_size=5320,
    shuffle_batches=True,
    instance_count=20000,
    max_seq_len=512,
)
valid_dataloader = DataLoader(valid_dataset, batch_size=285, collate_fn=valid_dataset.collate_fn) # NO HACER SHUFFLE!
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#model = SimpleGru(embedding_dim=768, hidden_dim=768, num_layers=2, bidirectional=True)
model = build_finetuning_model(
    model_name="bert-base-multilingual-cased",
    num_ini_punct_classes=2,
    num_fin_punct_classes=4,
    num_cap_classes=4,
    dropout=0.1,
    softmax_outputs=False,
    freeze_encoder=False
)

optimizer = torch.optim.Adam(model.parameters())

"""
En el script dataset_statistics.py calculamos las distribuciones de clases en el dataset de entrenamiento. La función de
pérdida de este modelo es una suma de las CrossEntropyLoss de cada una de las tres tareas (puntuación inicial, puntuación final y capitalización).

En cada tarea se toma la CrossEntropyLoss con 1-f siendo f la frecuencia de la clase en el dataset de entrenamiento, para compensar el desbalance de clases.
La función de pérdida total es la suma de las tres pérdidas.
"""
weight_initial_dict = {0: 1 - 0.9771971264600966, 1: 1 - 0.022802873539903384}
criterion_initial_loss = nn.CrossEntropyLoss(weight=torch.tensor([weight_initial_dict[i] for i in range(len(weight_initial_dict))], device=device))

weight_final_dict = {0: 1 - 0.8280982251374712, 2: 1 - 0.10794067714022632, 1: 1 - 0.03888745761118471, 3: 1 - 0.025073640111117713}
criterion_final_loss = nn.CrossEntropyLoss(weight=torch.tensor([weight_final_dict[i] for i in range(len(weight_final_dict))], device=device))

weight_capitalization_dict = {1: 1 - 0.22320756668702058, 0: 1 - 0.7535292389077267, 2: 1 - 0.013207690936229767, 3: 1 - 0.010055503469022951}
criterion_capitalization_loss = nn.CrossEntropyLoss(weight=torch.tensor([weight_capitalization_dict[i] for i in range(len(weight_capitalization_dict))], device=device))

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

model.to(device)
print(device)
# Esto guarda un log de entrenamiento en un archivo csv y guarda una copia de los pesos del modelo en después de cada epoch.
history = training_loop(model, device, train_dataloader, valid_dataloader, criterion_initial_loss, criterion_final_loss, criterion_capitalization_loss, optimizer, num_epochs=200)
