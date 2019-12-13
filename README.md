## Análise de Sentimento em Reviews de Filmes

### Deep Learning II - Especialização em Ciência de Dados - PUCRS/2019

Gibson Weinert, Luciano Gonçalves, João Paulo Medeiros

### Dataset

Para este trabalho foi escolhido o dataset [Kaggle - Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data),
o qual consiste em reviews de filmes do site Rotten Tomatoes, classificadas de acordo com o sentimento do autor do review, indo de negativo a positivo, passando por neutro, numa escala de zero a quatro.

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

Notou-se na [análise dos dados](dataset-analysis.ipynb) que os mesmos não estão seguindo uma distribuição uniforme com relação à classes, mas sim uma distribuição semelhante à normal.

### Modelo

Nossa implementação partiu do exemplo apresentado em aula, adaptado ao nosso dataset, e com ajustes na tentativa de melhorar a performance do classificador. Foi necessário gerar o vocabulário a ser usado pelo classificador, a partir do dataset completo.

Como no exemplo dado em aula, para a classificação dos reviews foram utilizadas redes neurais recorrentes, em especial a Long Short Term Memory (**LSTM**) bidirecional, seguidas de um **average pooling** e uma ou mais camadas **totalmente conectadas**.

| Modelo |
|:--:|
| RNN (LSTM, GRU) |
| AVG Pool |
| FC |

A função de custo utilizada foi a entropia cruzada (**CrossEntropyLoss**), e o otimizador utilizado foi o **Adam**. Para variar a taxa de aprendizado (LR) foi utilizado o **StepLR**, reduzindo a taxa a 1/10 da anterior a cada *lr_step* épocas.

### Experimentos de treino

Foram realizados diversos experimentos, onde alguns foram selecionados e estão listados na tabela abaixo. Lembrando que, por limitações de tempo e recursos computacionais, não foi possível explorar mais opções nesta tarefa de classificação, nem mesmo outros modelos alternativos além dos apresentados.

Nos experimentos de 1 a 5 notou-se sempre um rápido overfitting no conjunto de treino, resultando numa acurácia máxima em torno de 65% na avaliação. A redução da taxa de aprendizado e aumento no número de épocas não surtiu efeito neste comportamento, nem mesmo a adição de dropout, aumento das camadas da RNN, aumento das cadamas FC ou a redução do tamanho do batch.
O que se nota é que o classificador aprendeu a distribuição das classes, semelhante à normal, e tem grandes dificuldades em classificar corretamente as classes menos frequentes do dataset. 

Para experimento 6 removemos do dataset as multiplas frases de cada sentença, deixando somente as sentenças completas. Desta forma a distribuição das classes ficou menos parecida com a normal, e mais próxima da uniforma, embora ainda longe da uniforme propriamente dita. Também aumentamos o número de épocas e reduzimos a taxa de aprendizado inicial.
Porém o resultado foi uma acurácia máxima no treinamento em torno de 35%, reflexo da nova distribuição das classes, num comportamento semelhante ao do dataset original.

*Clique no número da versão abaixo para abrir o respectivo jupyter notebook.*

| Versão | Épocas | LR Inicial | LR Step | RNN | Camadas RNN | Dropout | Camadas FC | Máxima acurácia no treino |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| [1](sentiment-analysis_v01.ipynb) | 10 | 1e-2 | 3 | LSTM | 1 | 0 | 1 | 65.65% |
| [2](sentiment-analysis_v02.ipynb) | 10 | 1e-2 | 3 | LSTM | 1 | **0.2** | 1 | 65.41% |
| [3](sentiment-analysis_v03.ipynb) | 12 | 1e-2 | 3 | LSTM | **3** | 0.2 | 1 | 65.59% |
| [4](sentiment-analysis_v04.ipynb) | 12 | 1e-2 | **4** | LSTM | 3 | 0.2 | **2** | 65.33% |
| [5](sentiment-analysis_v05.ipynb) | 12 | 1e-2 | 4 | **GRU** | 3 | 0.2 | 2 | 65.37% |
| *[6](sentiment-analysis_v06.ipynb)* | *50* | *1e-3* | *3* | *LSTM* | *3* | *0.2* | *2* | *35.24%* |

### Teste

Como as medidas de acurácia não variaram muito de uma versão para a outra, para o teste foi escolhida a versão 4, por possuir mais layers tanto no LSTM quanto na FC, o que em teoria tornaria ela mais robusta do que as anteriores por trazer maior poder de abstração e poder de classificação não linear, respectivamente.

Como era de se esperar, o resultado no teste foi apenas um pouco pior que a valização, com acurácia de 65,09% e matriz de confusão semelhante.

### Conclusão

Neste trabalho foram exploradas algumas alternativas na tentativa de otimizar um classificador de sentimentos em textos usando RNNs (LSTM e GRU bidirecionais) no dataset Rotten Tomatoes.

Na etapa de treino e validação a melhor acurácia atingida foi de 65,65%. Nota-se pelas matrizes de confusão que os classificadores aprenderam a distribuição dos dados, que era semelhante à distribuição normal, e dessa forma não conseguiam ter boa performance na classificação de classes com menos instâncias no dataset.

No teste o desempenho foi semelhante ao da validação, ficando um pouquinho pior.

Por limitações de tempo e recursos computacionais, não foi possível explorar mais opções nesta tarefa de classificação, nem mesmo outros modelos alternativos além dos apresentados. Fica a sugestão para trabalhos futuros de usar outros modelos, como redes convolucionais, por exemplo.
