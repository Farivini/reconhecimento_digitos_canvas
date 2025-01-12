# Reconhecimento de Dígitos MNIST em um Canvas


https://reconhecimentodigitoscanvas-hx9nd3kduzjf2rdeonwccw.streamlit.app/


Este projeto demonstra como treinar uma **Rede Neural** simples para reconhecer dígitos (0 a 9) usando o famoso dataset **MNIST** e uma aplicação **Streamlit** que permite desenhar os dígitos diretamente em um **canvas**.

## Visão Geral

1. **Carregamento do Dataset**: O MNIST contém dígitos manuscritos em escala de cinza (0 a 9), cada imagem com **28×28** pixels.  
2. **Treino da Rede Neural**: Após carregar o conjunto de dados, podemos:
   - Separar uma parte para **validação**.
   - **Normalizar** os valores de pixel (0–255 → 0.0–1.0).
   - Treinar o modelo por algumas épocas (*epochs*).
3. **Avaliação**: Verificamos o quão bem o modelo se sai em dados que ele ainda não viu (conjunto de teste).
4. **Desenho em Canvas**: Podemos desenhar qualquer dígito (0 a 9) em um quadrado de 280×280. O aplicativo converte o desenho em **28×28**, inverte as cores (pois o MNIST possui dígito claro em fundo escuro) e pede ao modelo para adivinhar o dígito.
5. **Salvar/Carregar Pesos**: Para não precisar retreinar sempre, salvamos os pesos da rede em um arquivo `.weights.h5` e podemos carregá-los posteriormente.

## Principais Ferramentas

- **Python 3**
- **Streamlit** — interface web simples.
- **TensorFlow/Keras** — para construir e treinar a rede neural.
- **Matplotlib, Numpy, Pandas** — suporte a gráficos, arrays e manipulação de dados.
- **streamlit-drawable-canvas** — permite desenhar diretamente no app Streamlit.
- **PIL (Pillow)** — para manipular imagens.

## Executando o Projeto

1. **Instale as dependências** (caso já não as tenha instaladas) e rode:
   ```bash
   pip install streamlit tensorflow keras matplotlib pandas pillow streamlit-drawable-canvas
   streamlit run app_mnist_canvas.py

## Estrutura Principal do Projeto

### `app_mnist_canvas.py`
**Contém:**
- Carregamento do MNIST.
- Funções de split, normalização, treino e avaliação do modelo.
- Área de desenho (canvas) para a predição.
- Opções de salvar/carregar os pesos do modelo.
- Barra de progresso durante o treino, para feedback ao usuário.

---

## Como Funciona o Canvas?

- Ao final do aplicativo, existe uma **área de desenho** de 280×280 pixels.
- Você desenha seu dígito (0 a 9) em **preto**, enquanto o fundo permanece branco.
- Ao clicar em **“Predizer Dígito”**:
  1. O desenho é convertido para **escala de cinza** e reduzido para **28×28** pixels.
  2. As cores são **invertidas** (fundo escuro, dígito claro), pois o MNIST “original” é assim.
  3. A imagem é **normalizada** (dividida por 255.0) para ficar entre 0 e 1.
  4. A rede neural faz a **predição**, retornando a probabilidade de ser cada um dos dígitos (0 a 9).

---

## Autor

**Vinicius Farineli Freire**  
[LinkedIn: Vinicius Farineli Freire](https://www.linkedin.com/in/viniciusfarinelifreire/)
