import numpy as np
import pandas as pd
import torch
from TP2.tokenizer import get_tokenizer_and_bert_embeddings
from TP2.dataloader import idx_to_punct_inicial, idx_to_punct_final
from TP2.tokenizer import Capitalizaciones

def recontruccion_texto_catedra(texto_catedra,modelo_entrenado,device):
  #asumo que texto_catedra es un dataframe con columnas: instancia_id, token_id, token
  #te llega  instancia_id,token_id, token
  tokenizer, embedding_matrix = get_tokenizer_and_bert_embeddings()
  modelo_entrenado.to(device)
  modelo_entrenado.eval()
  # como el df es token_id, quiero agregar una columna embedding al df por cada token id
  texto_catedra['embedding'] = texto_catedra['token_id'].apply(lambda x: embedding_matrix[x]) #nose si a este paso hay que ahcerle algun mean por cada token de una palabara ayuda jorgpt
  prediccion_porinstancia=[]
  for numero_grupo_intancia, instancia in texto_catedra.groupby(texto_catedra['instancia_id'], sort=False):
    embeddings_de_oracion = np.stack(instancia['embedding'].to_list()) #tengo los embeddings de la instancia
    with torch.no_grad():
         # asumo que anda?
        salida = modelo_entrenado(torch.tensor(embeddings_de_oracion).to(device))
        salida_punt_ini, salida_punt_fin, salida_cap = salida
        for linea in range(instancia.shape[0]): #esto lo hago asumiendo que 1° token  y salida de la preddicion de 1° token coinciden
                prediccion_porinstancia.append({
                    "instancia_id": instancia.iloc[linea]['instancia_id'],
                    "token_id": instancia.iloc[linea]['token_id'],
                    "token": instancia.iloc[linea]['token'],
                    "punt_inicial": idx_to_punct_inicial[torch.argmax(salida_punt_ini[linea]).item()],
                    "punt_final": idx_to_punct_final[torch.argmax(salida_punt_fin[linea]).item()],
                    "capitalización": Capitalizaciones.from_int(torch.argmax(salida_cap[linea]).item()).value,
                })
           

  return pd.DataFrame(prediccion_porinstancia)
    

