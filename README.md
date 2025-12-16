# TCC Software 2025

Este repositório contém os códigos utilizados para a parte de Software no Trabalho de Conclusão de Curso (TCC) com o tema: Monitoramento Inteligente da Demanda de Transporte Público na Cidade Universitária da USP

## Descrição dos Componentes
🔹 codigo_yolo.py

Script principal responsável por:
- Carregar o modelo YOLOv8 (pré-treinado)
- Recuperar imagens do S3
- Processar imagens
- Detectar/contar pessoas nas imagens
- Salvar informações/resultados no Amazon DynamoDB

🔹 Notebook Visualizacao_tcc.ipynb

Este notebook contém:
- Visualizações dos resultados
  -  Gráficos comparativos entre as contagens reais e as preditas pelo modelo
  -  Gráfico com a tensão da bateria ao longo do tempo (verificação do seu descarregamento)
- Cálculo da métrica MAE (Mean Absolute Error) para avaliação do desempenho
