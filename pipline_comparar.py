#quieor hacer una pipeline para comparar los modelos.
from sklearn.metrics import f1_score
import torch


def predicciones_del_modelo(modelo, dataloader, nombre_modelo):
    modelo.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    modelo.to(device)
    punct_ini_pred, punct_ini_ground_truth = [], []
    punct_fin_pred, punct_fin_ground_truth = [], []
    cap_pred, cap_ground_truth = [], []
    criterion = torch.nn.CrossEntropyLoss()
    total_samples = 0
    running_loss = 0.0
    with torch.no_grad():
        for embedding_group, punt_inicial_group, punt_final_group, capitalizacion_group in dataloader:
            for embedding, true_punt_inicial_idx, true_punt_final_idx, true_capitalizacion_idx in zip(embedding_group,
                                                                                                      punt_inicial_group,
                                                                                                      punt_final_group,
                                                                                                      capitalizacion_group):
                # Para cada "oración" (instancia) en el batch
                total_samples += 1
                instance_length = embedding.size(0)
                embedding = embedding.to(device).unsqueeze(0)

                true_punt_inicial_idx = true_punt_inicial_idx.to(device)
                true_punt_final_idx = true_punt_final_idx.to(device)
                true_capitalizacion_idx = true_capitalizacion_idx.to(device)

                out_punct_ini, out_punct_fin, out_cap = modelo(embedding)  # forward pass - predicciones de las puntuaciones y capitalización de la instacia
                out_punct_ini = out_punct_ini.permute(0, 2, 1)
                out_punct_fin = out_punct_fin.permute(0, 2, 1)
                out_cap = out_cap.permute(0, 2, 1)

                running_loss += criterion(out_punct_ini, true_punt_inicial_idx)
                running_loss += criterion(out_punct_fin, true_punt_final_idx)
                running_loss += criterion(out_cap, true_capitalizacion_idx)
                _, pred_punct_ini = torch.max(out_punct_ini, dim=1)
                _, pred_punct_fin = torch.max(out_punct_fin, dim=1)
                _, pred_cap = torch.max(out_cap, dim=1)

                punct_ini_pred.extend(pred_punct_ini.cpu().squeeze(0).tolist())
                punct_ini_ground_truth.extend(true_punt_inicial_idx.cpu().squeeze(0).tolist())
                punct_fin_pred.extend(pred_punct_fin.cpu().squeeze(0).tolist())
                punct_fin_ground_truth.extend(true_punt_final_idx.cpu().squeeze(0).tolist())
                cap_pred.extend(pred_cap.cpu().squeeze(0).tolist())
                cap_ground_truth.extend(true_capitalizacion_idx.cpu().squeeze(0).tolist())
    avcg_loss = running_loss / total_samples
    f1_punt_ini = f1_score(punct_ini_pred, punct_ini_ground_truth, zero_division=0, average="macro")
    f1_punt_fin = f1_score(punct_fin_pred, punct_fin_ground_truth, average="macro", labels=[0,1,2,3], zero_division=0)
    f1_cap = f1_score(cap_pred, cap_ground_truth, average="macro", labels=[0,1,2,3], zero_division=0)
    return {
        
        "nombre_modelo": nombre_modelo,
        "avg_loss": avcg_loss.to("cpu").item(),  
        "f1_punct_ini": f1_punt_ini,
        "f1_punct_fin": f1_punt_fin,
        "f1_cap": f1_cap,
        "punct_ini_pred": punct_ini_pred,
        "punct_ini_ground_truth": punct_ini_ground_truth,
        "punct_fin_pred": punct_fin_pred,
        "punct_fin_ground_truth": punct_fin_ground_truth,
        "cap_pred": cap_pred,
        "cap_ground_truth": cap_ground_truth
    }   

def resultados_mejores_modelos(dataloader,modelos):
    #cargar mejor configuracion de modelo
    resultados = {}
   
    for nombre_modelo, modelo in modelos.items():
        modelo.load_state_dict(torch.load(f"TP2\models\DictMejores\{nombre_modelo}_best_model.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        modelo.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        modelo.to(device)
        resultado = predicciones_del_modelo(modelo, dataloader,nombre_modelo)
        
        resultados[nombre_modelo] = resultado
           
    return resultados

def print_tabla_modelos(resultados):
  for nombre_modelo, metricas in resultados.items():
        
      
        print(f"Resultados para el modelo {nombre_modelo}:")
        print(f"  Pérdida: {metricas["avg_loss"]}")
        print(f"  F1 Puntuación Inicial: {metricas["f1_punct_ini"]}")
        print(f"  F1 Puntuación Final: {metricas["f1_punct_fin"]}")
        print(f"  F1 Capitalización: {metricas["f1_cap"]}")

def comparar_modelos(dataset_path=None, modelos = None):


    from torch.utils.data import DataLoader
    from TP2.dataloader import TextDataset

    comparacion_dataseth = TextDataset( #esto no se bien
   
    base_path=dataset_path,
    parquet_file_subpath="valid_tokens_batch_",
    batch_size=2048,
    instance_count=31875,
    max_seq_len=512
    
)
    comparacion_dataloader = DataLoader(comparacion_dataseth, batch_size=600, collate_fn=comparacion_dataseth.collate_fn) # NO HACER SHUFFLE! TODO explicar por qué no shuffle

    #obtener resultados de los modelos
    resultados = resultados_mejores_modelos(comparacion_dataloader,modelos)
    
    #imprimir resultados
    print_tabla_modelos(resultados)
    return resultados
