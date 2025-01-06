# Reconhecimento de Dígitos com MNIST e Canvas

Este projeto demonstra como **treinar** uma rede neural simples para **reconhecer dígitos** (0 a 9) usando o famoso dataset **MNIST**. A aplicação é feita com **Streamlit** e inclui uma área de desenho (*canvas*) para que você possa **desenhar** seu próprio dígito e testar a predição na hora.

## Funcionalidades

1. **Carregar e Explorar o Dataset MNIST**  
   - O MNIST contém 70.000 imagens de dígitos manuscritos (60.000 para treino e 10.000 para teste), cada imagem com 28×28 pixels.

2. **Treinar Modelo**  
   - Dividir parte do dataset para validação.  
   - Normalizar os dados (0-255 → 0.0-1.0).  
   - Escolher quantas épocas (epochs) e qual otimizador (SGD, Adam etc.).  
   - Acompanhar o **progresso** do treino com uma **barra de progresso**.

3. **Avaliar no Conjunto de Teste**  
   - Ver como o modelo performa em dados que não foram usados no treino.

4. **Canvas para Desenhar Dígitos**  
   - Desenhe em um quadro de 280×280 pixels.  
   - O app converte para 28×28, **inverte as cores** (já que o MNIST tem fundo escuro) e prediz o dígito.

5. **Salvar e Carregar Pesos**  
   - Salvar apenas os pesos do modelo em um arquivo `.weights.h5`.  
   - Evita precisar treinar de novo toda vez.

## Como Executar

1. **Clone** este repositório ou baixe os arquivos.

2. **Instale as dependências** (se possuir um arquivo `requirements.txt`):
   ```bash
   pip install -r requirements.txt
