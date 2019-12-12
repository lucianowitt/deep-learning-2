## Análise de Sentimento em Reviews de Filmes

### Deep Learning II - Especialização em Ciência de Dados - PUCRS/2019

Gibson Weinert, Luciano Gonçalves, João Paulo Medeiros

### Dataset

Para este trabalho foi escolhido o dataset [Kaggle - Rotten Tomatoes](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data),
o qual consiste em reviews de filmes classificadas de acordo com o sentimento do autor do review, indo de negativo a positivo, passando por
neutro, numa escala de zero a quatro.

**Classes**

* 0 - negative
* 1 - somewhat negative
* 2 - neutral
* 3 - somewhat positive
* 4 - positive

### Análise dos dados

O dataset original está dividido nos conjuntos de treino e de teste. Porém ele foi concebido para um desafio, então o conjunto de teste não
possui labels. Como o objetivo deste trabalho não é participar de tal desafio, o conjunto de teste foi descartado, e o conjunto de treino original
foi dividido nos conjuntos de treino, validação e teste, sendo 20% do total para teste, e 20% do restante para validação.

| Conjunto | Registros |
|---|--:|
| Treino | 99878 |
| Validação | 24970 |
| Teste | 31212 |
| *Total* | *156060* |

Cada registro do dataset original possui quatro colunas: um identificador de frase, um identificador de sentença, a frase (texto) e a label do sentimento.
Isso porque cada sentença da review foi dividida em "frases", de forma que uma frase isolada pode ser neutra, por exemplo, enquanto a sentença completa é
negativa, pois outra frase da mesma sentença possui o fator que define o sentimento. Desta forma supostamente seria possível que os algoritmos que
participassem do desafio aprendessem a diferenciar as partes das sentenças que definem os sentimentos.

Para nosso trabalho descartamos os identificadores de sentença e frase e utilizamos apenas os textos (coluna "Phrase") e labels (coluna "Sentiment") dos registros,
renomeadas para "X" e "y", respectivamente.

### Modelo

Nossa implementação partiu do exemplo apresentado em aula, adaptado ao nosso dataset, e com ajustes para melhorar a performance do classificador.
Foi necessário gerar o vocabulário a ser usado pelo classificador, a partir do dataset completo.
Como no exemplo dado em aula, para a classificação dos reviews foi utilizada uma Long Short Term Memory (LSTM) bidirecional.

### Experimentos de treino

Foram realizados vários experimentos, e os cinco mais relevantes serão apresentados neste trabalho, numa sequência de ajustes nas configurações e hiperparâmetros que
levaram ao resultado final.
Lembrando que, por limitações de tempo e recursos computacionais, não foram exploradas mais opções de otimização deste modelo, nem mesmo modelos alternativos.
Este trabalho não tem por objetivo encontrar a melhor solução para o problema, apenas exercitar os conhecimentos adquiridos durante o curso.

| Versão | Épocas | LR inicial | LR step | Layers | Máxima acurácia no treino |
|:--:|:--:|:--:|:--:|:--:|:--:|
| [1](sentiment-analysis_v01.ipynb#Resultados-do-treino) | 10 | 1e-2 | 3 | 1 | 77.39% |
| [2](sentiment-analysis_v01.ipynb#Resultados-do-treino) | 20 | 1e-2 | **5** | 1 | 83.54% |
| [3](sentiment-analysis_v01.ipynb#Resultados-do-treino) | 20 | 1e-2 | 5 | **3** | 86.29% |
| [4](sentiment-analysis_v01.ipynb#Resultados-do-treino) | 20 | 1e-2 | 5 | **5** | 84.90% |
| [5](sentiment-analysis_v01.ipynb#Resultados-do-treino) | **25** | 1e-2 | **7** | 5 | 90.04% |

### Teste

*Pendente: descrição do teste*

### Conclusão

Neste trabalho foram exploradas algumas alternativas na otimização de um classificador de sentimentos de textos usando uma LSTM bidirecional no dataset Rotten Tomatoes.
Foi atingida em tempo de treino a acurácia de 90.04%.

*Pendente: conclusão sobre o teste*
