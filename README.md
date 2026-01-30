# dspy_ai_learning

Este projeto é um repositório dedicado ao estudo e aprendizagem da framework DSPy.AI. O objetivo principal é explorar como a programação declarativa de modelos de linguagem (LLMs) pode substituir o prompt engineering manual por processos de otimização sistemáticos e programáticos.

## Estrutura do Projeto

O repositório está organizado da seguinte forma para facilitar o fluxo de desenvolvimento e experimentação:

- data/: Pasta que contém os ficheiros de dados (CSV, JSON, etc.) utilizados para treinar e avaliar os programas.
- signatures/: Definições das assinaturas do DSPy, que especificam o comportamento pretendido para as entradas e saídas de cada módulo.
- modules/: Implementação de programas DSPy complexos que utilizam uma ou mais assinaturas.
- optimizers/: Scripts dedicados à configuração e execução de teleprompters (otimizadores) para refinar os prompts.
- config/: Configurações de ambiente e definição dos modelos de linguagem (LM) e modelos de recuperação (RM).
- notebooks/: Demonstrações interativas e testes rápidos de conceitos aprendidos.
- requirements.txt: Lista de dependências necessárias para a execução do projeto.

## Tecnologias Envolvidas

As seguintes tecnologias e bibliotecas formam a base deste projeto:

- Python: Linguagem de programação principal.
- DSPy: Framework para programação e otimização de sistemas baseados em LLM.
- Modelos de Linguagem: Utilização de backends como OpenAI, Anthropic ou Ollama para processamento.
- Dotenv: Gestão de variáveis de ambiente e chaves de API.
- Pandas: Manipulação de dados para os conjuntos de treino e teste.

## Conjunto de Dados (Dataset)

Para o processo de aprendizagem, o projeto utiliza conjuntos de dados estruturados no formato dspy.Example. O foco principal recai sobre:

- Exemplos de Treino (Trainset): Uma pequena amostra de dados que serve para o otimizador aprender o padrão de resposta esperado.
- Exemplos de Validação (Devset): Utilizados para monitorizar o desempenho do modelo durante a otimização e evitar o sobreajuste (overfitting).
- Métricas: Definição de funções programáticas que avaliam se a saída do modelo está correta, permitindo que o DSPy quantifique a qualidade das respostas.

## Otimizadores (Teleprompters)

Os otimizadores são o núcleo do DSPy e são utilizados neste projeto para automatizar a criação de prompts eficazes. Os principais tipos abordados são:

- BootstrapFewShot: Este otimizador gera automaticamente exemplos de poucos disparos (few-shot) para incluir no prompt. Ele executa o programa e seleciona as instâncias que passam na métrica definida para guiar o modelo.
- BootstrapFewShotWithRandomSearch: Uma extensão que realiza múltiplas iterações de busca aleatória sobre as combinações de exemplos gerados, selecionando a combinação que obtém a melhor pontuação na métrica de avaliação.
- COPRO (Chain of Thought Optimization): Focado em otimizar as instruções textuais e os prefixos das assinaturas, refinando a forma como a tarefa é explicada ao modelo.

Este repositório serve como um guia prático para transformar interações com inteligência artificial em fluxos de engenharia reprodutíveis.


## Comandos auxiliares:

Para rodar o projeto python:
- python src/app/main.py

Para rodar como um módulo( E
- python -m src.app.main

upgrade no dspy-ai:
- uv pip install "dspy-ai>=3.0.2"

Adentrar ao ambiente .vnev
- source .venv/bin/activated
 - python src/app/main.py

Para rodar como um módulo:
- python -m src.app.main

upgrade no dspy-ai:
- pip install "dspy-ai>=3.0.2"

Adentrar ao ambiente .venv
- source .venv/bin/activate

---
Feito com vontade por Airton Junior
OBS: Projeto desenvolvido no Ubuntu 24.04 através do WSL2
https://www.linkedin.com/in/airton-lira-junior-6b81a661/