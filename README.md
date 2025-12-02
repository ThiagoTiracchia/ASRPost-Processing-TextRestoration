# TP 2 - Predicción de puntuación y capitalización en texto normalizado

# TO-DO

- Elegir dataset definitivo.
- Entrenar con ese dataset la RNN Simple, probar darle pesos a las losses según la frecuencia de cada clase.
- Entrenar con RNN Bidireccional y probar lo mismo.
- Ver Random Forest o algun modelo clasico
- Ver caso de las Named Entities para la capitalización.
- Pipeline para evaluacion.
- Reconstrucción de predicciones en el formato pedido.

### Cosas que se dijeron en clase

- Si la palabra no está en el diccionario se puede dividir y buscar si está
- **Preprocesamiento**: Buscar un dataset y quitarle todos los signos de puntuación y mayúsculas.
	- Hay que usar como etiquetas los signos y mayúsculas
- **Embeddings**: relación entre palabras (e.g casa, departamento).
	- usamos embedding pre-entrenado (bert)

## Dataset

Compuesto por

- [Language-Independent Named Entity Recognition (I)](https://www.clips.uantwerpen.be/conll2002/ner/)
  - The data consists of two columns separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence. The first item on each line is a word and the second the named entity tag. The tags have the same format as in the chunking task: a B denotes the first item of a phrase and an I any non-initial word. There are four types of phrases: person names (PER), organizations (ORG), locations (LOC) and miscellaneous names (MISC).
- [Crowdsourced high-quality Argentinian Spanish speech data set](https://openslr.org/61/)
  - This dataset contains transcribed high-quality audio of random Spanish sentences recorded by volunteers in Buenos Aires, Argentina. The data set consists of wave files, and a TSV file (line_index.tsv). The file line_index.tsv contains a anonymized FileID and the transcription of audio in the file.
- [Spanish speech text - HuggingFace](https://huggingface.co/datasets/PereLluis13/spanish_speech_text)
- [Corpus de novelas hispanoamericanas del siglo XIX (conha19)](https://github.com/cligs/conha19)
## Papers

- [RNN Approaches to Text Normalization: A Challenge](https://arxiv.org/abs/1611.00068)
- [Punctuation Prediction Model for Conversational Speech](https://arxiv.org/pdf/1807.00543)
- [Punctuation prediction using a bidirectional recurrent neural network with part-of-speech tagging](https://www.researchgate.net/publication/322216246_Punctuation_prediction_using_a_bidirectional_recurrent_neural_network_with_part-of-speech_tagging)
- [Capitalization and Punctuation Restoration: a Survey](https://arxiv.org/pdf/2111.10746)

## Poster

- https://docs.google.com/presentation/d/1KKfwic0pXZl3AJhWZAJnoe0H6vVmby6-MAYacAQT3zg/edit?usp=sharing
