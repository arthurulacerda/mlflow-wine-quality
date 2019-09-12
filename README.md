# mlflow-wine-quality

Requisitos: ``mlflow 1.2.0`` e ``conda 4.7.10``

Contextualização
----------------

Estudo experimental realizado com MLflow, apresentado na Campus Party Goiás 2019.

O problema atacado seria a predição da qualidade de um vinho, baseado em suas características, no projeto, temos dois arquivos que treinam modelos de **regressão**, que sua predição é a qualidade de um vinho (valor numérico), que é o label original do dataset, e um arquivo que treina um modelo de **classificação**, onde se assume que um vinho pode ser 'bom' ou 'ruim', onde bons vinhos para treinamento, são aqueles que tinham qualidade maior ou igual a 7, e os demais são classificados como ruins.

Interface
---------
Inicializar::

	mlflow ui

Acessar em, http://localhost:5000

Executar projetos
-----------------

Para executar os projetos, o comando deverá ser executado no path em que se encontra o arquivo `MLproject`, que contém as definições de execução. 

Neste projeto, temos os pontos de entrada `main`, `huber`  e `classifier`.

Para executar o projeto com a entrada `main`, basta executar a linha abaixo, substituíndo `<path>` pelo diretório em que se encontra o arquivo `MLproject`::

	mlflow run <path>

Também é possível fazer o treinamento do modelo, acessando diretamente o link do GitHub no lugar do `<path>`, dessa forma não é necessário que você tenha uma clonagem do diretório em sua máquina::

	mlflow run https://github.com/arthurulacerda/mlflow-wine-quality

Para executar outras entradas, basta utilizar o parâmetro `-e <nome_da_entrada>` antes de especificar o `<path>`, como no exemplo abaixo::

	mlflow run -e huber <path>

À cada execução, é gerada uma instância de um `Model` no diretório `mlruns`, os parâmetros, artefatos, modelos e métricas catalogadas podem ser verificadas na interface. Cada uma dessas execuções retorna um ID, estará presente no diretório `mlruns`, ele também pode ser identificado na interface, ao acessar um treinamento específico.

Passagem de parâmetros
----------------------

Para passar parâmetros à execução de um projeto, após o comando, especifique os parâmetros presentes em `MLproject` com `-P <nome_do_parametro>=valor`

Exemplos::

	mlflow run https://github.com/arthurulacerda/mlflow-wine-quality -P alpha=0.47 -P l1_ratio=0.2

	mlflow run -e huber <path> -P epsilon=1

Servir o modelo
---------------

Para servir um modelo, escolha algum dos IDs das execuções geradas, e execute o comando abaixo para servir o modelo:

	mlflow models serve -m mlruns/0/<ID>/artifacts/model/ --port 5001

Obs: A porta foi especificada em 5001, pois por padrão ela executa na 5000, em que já está sendo executada a interface.

Nesse ponto, a porta 5001 fica aguardando chamadas REST para realizar predições do modelo. para as entradas `main` e `huber`, que são modelos de regressão, podemos testar utilizando o seguinte comando::

	curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[6.0, 0.29, 0.41, 10.8, 0.048, 55.0, 149.0, 0.9937, 3.09, 0.59, 10.966667]]}' http://127.0.0.1:5001/invocations

que retornará algo semelhante à:

	[6.197856454567]

para a entrada `classifier`, podemos testar utilizando o comando abaixo::

	curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[-0.96860107,  0.16580424,  0.55633228,  0.97056386,  0.09408896,  1.03266429,  0.20843976, -0.05401888, -0.61083071,  0.85930445,  0.34699202]]}' http://127.0.0.1:5001/invocations

que retornará algo semelhante à:

	[0]

Obs: As duas chamadas se diferem, pois no treinamento do classificador, foi feito um pré processamento, que pode ser verificado em `clf-train.py`. Os dois contextos também se diferem na resposta dada em predição, isso ocorre pois na entrada `main` e `huber` o retorno é uma nota de predição para o vinho, já em `classifier`, ele retorna `1` para quando o vinho é considerado como 'bom' e `0` para um vinho ruim.

Predição em Batch
-----------------

É possível realizar predições em batch por um arquivo CSV, não é necessário que o modelo esteja sendo servido, basta executar o comando abaixo::

	mlflow models predict -m mlruns/0/<ID>/artifacts/model -i <path>/predict.csv -t 'csv'

Experimentos
------------

Para aumentar o nível de organização, é possível criar diferentes experimentos, como por exemplo, podemos criar um experimento diferente para executar nossa entrada `classifier`, já que é um outro contexto de análise, e as métricas em relação aos outros modelos não são comparáveis, mas se trata de uma abordagem nova. Para criar um novo experimento, podemos executar::

	mlflow experiments create -n classificacao

Desta forma, é possível executar o projeto, referenciando um experimento específico, utilizando o parâmetro `--experiment-name <nome_do_experimento`, como por exemplo::

	mlflow run -e classifier <path> --experiment-name classifier

Através da interace, é possível verificar o novo experimento.

Também notamos que no diretório `mlflow` foi criado um novo subdiretório `mlflow/1`, que armazenará os modelos do novo experimento. Por padrão ele armazena em `mlflow/0`, já que 0 é referente ao ID do experimento Default.

Dito isso, podemos servir os modelos gerados nesse experimento, a partir do novo ID do experimento gerado::

	mlflow models serve -m mlruns/1/<ID>/artifacts/model/ --port 5001

Para listar os experimentos existentes, e verificar seus respectivos IDs::

	mlflow experiments list

Criar imagem docker / servir modelo pela imagem
-----------------------------------------------

O MLflow permite criar uma imagem de máquina docker que servirá o modelo a partir do seguinte comando::

	mlflow models build-docker -m mlruns/<ID_EXPERIMENTO>/<ID_MODEL>/artifacts/model -n <nome_da_imagem>

A criação da imagem toma um certo tempo, requer um espaço de armazenamento durante a montagem. Ao fim do procedimento, a imagem escutará requisições na porta 8080 da imagem.

Para servir, aponte a porta 8080 da imagem à uma porta específica da sua máquina. Para este passo, é necessário ter `docker` instalado. Ao executar o seguinte comando, será possível servir o modelo, via máquina docker na porta 5001::

	docker run -p 5001:8080 <nome_da_imagem>


Documentação
------------

Para mais informações, acesse a documentação oficial do MLflow: https://mlflow.org/docs/latest/index.html.