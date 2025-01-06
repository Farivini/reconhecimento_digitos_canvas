import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import time
import cv2
# Keras / TensorFlow
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense

# Para desenhar no canvas
from streamlit_drawable_canvas import st_canvas

# Ajuste opcional de layout (fontes, etc.)
st.set_page_config(layout="centered")
st.markdown("""
<style>
/* Fonte um pouco menor e títulos com espaçamento menor */
body {
    font-size: 0.9rem;
    background-color: #F9FAFB; /* leve cinza */
    color: #333;
}
h1, h2, h3, h4 {
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}
.block-container {
    background-color: #FFFFFF; /* branco */
    padding: 2rem 2rem 2rem 2rem;
    border-radius: 8px;
    box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# Autor / LinkedIn
# -------------------------------------------------------------------------------------------------
st.sidebar.markdown("""
**Autor**: Vinicius Farineli Freire  
[LinkedIn](https://www.linkedin.com/in/viniciusfarinelifreire/)                   
[Page Resumo habilidades](https://science-showcase-landing.lovable.app)
[Resumo Streamlit cv](https://resumecv-viniciusfarineli.streamlit.app)
""")

# Título principal
st.title("Classificação de Dígitos (MNIST) com Canvas")

st.write("""
Seja bem-vindo(a)!  
Neste aplicativo, você pode:

1. **Carregar** o dataset MNIST (dígitos 0-9).
2. **Visualizar** algumas imagens para entender o dataset.
3. **Treinar** um modelo (Rede Neural) para reconhecer esses dígitos.
4. **Ver** como ficou a performance no conjunto de teste.
5. **Desenhar** um dígito na área de desenho (canvas) e checar a **predição** do modelo.

---  
""")

# -------------------------------------------------------------------------------------------------
# Callback para a barra de progresso durante o treino
# -------------------------------------------------------------------------------------------------
class ProgressBarCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Atualiza a barra de progresso ao final de cada época
        current_progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(current_progress)

# -------------------------------------------------------------------------------------------------
# Seção 1: Carregar e Explorar MNIST
# -------------------------------------------------------------------------------------------------
st.subheader("1) Carregar e Explorar o Dataset MNIST")

if "mnist_loaded" not in st.session_state:
    st.session_state["mnist_loaded"] = False

col_load, col_show = st.columns(2)

with col_load:
    st.write("Clique no botão para baixar o MNIST.")
    if st.button("Carregar MNIST"):
        (x_original, y_original), (x_teste, y_teste) = mnist.load_data()
        
        st.session_state["x_original"] = x_original
        st.session_state["y_original"] = y_original
        st.session_state["x_teste"] = x_teste
        st.session_state["y_teste"] = y_teste
        st.session_state["mnist_loaded"] = True

        st.success("MNIST carregado com sucesso!")
        st.write(f"**x_original**: {x_original.shape}, **y_original**: {y_original.shape}")
        st.write(f"**x_teste**: {x_teste.shape}, **y_teste**: {y_teste.shape}")

with col_show:
    if st.session_state["mnist_loaded"]:
        st.write("Visualizar algumas imagens do MNIST (dígitos manuscritos):")
        num_images = st.slider("Quantas imagens mostrar?", 1, 10, 5)

        fig = plt.figure(figsize=(6,6))
        for i in range(num_images):
            plt.subplot(1, num_images, i+1)
            plt.imshow(st.session_state["x_original"][i], cmap='gray')
            plt.title(f"Rótulo: {st.session_state['y_original'][i]}")
            plt.axis('off')
        st.pyplot(fig)
    else:
        st.info("Carregue o MNIST para ver algumas imagens.")


# -------------------------------------------------------------------------------------------------
# Seção 2: Pré-processar e Treinar Modelo
# -------------------------------------------------------------------------------------------------
st.subheader("2) Pré-processar Dados e Treinar Modelo")
st.write("""
- **Dividir** em treino e validação (uma parte pequena para ver se o modelo está mesmo aprendendo).
- **Normalizar** (valores 0-255 viram 0.0-1.0).
- **Treinar** um modelo de Rede Neural simples (2 camadas densas + 1 camada de saída).
""")

if "modelo" not in st.session_state:
    st.session_state["modelo"] = None

if st.session_state["mnist_loaded"]:
    if st.button("Criar Split (Treino/Validação) e Normalizar"):
        x_original = st.session_state["x_original"]
        y_original = st.session_state["y_original"]

        # Normaliza (0-1)
        x_original = x_original / 255.0

        # Separa 10% (~6000) para validação
        x_validacao, x_treino = x_original[:6000], x_original[6000:]
        y_validacao, y_treino = y_original[:6000], y_original[6000:]

        # Guardar nos estados
        st.session_state["x_treino"] = x_treino
        st.session_state["y_treino"] = y_treino
        st.session_state["x_validacao"] = x_validacao
        st.session_state["y_validacao"] = y_validacao

        st.success("Divisão e normalização concluídas.")
        st.write(f"**x_treino**: {x_treino.shape}, **y_treino**: {y_treino.shape}")
        st.write(f"**x_validacao**: {x_validacao.shape}, **y_validacao**: {y_validacao.shape}")

    st.write("**Selecione alguns parâmetros de treino**:")
    epochs = st.slider("Épocas (quantas vezes o modelo vê todos os dados)?", 1, 20, 3)
    optimizer_choice = st.selectbox("Otimizador (como a rede ajusta os pesos):", ["sgd", "adam"])

    if st.button("Treinar Modelo"):
        if "x_treino" in st.session_state:
            x_treino = st.session_state["x_treino"]
            y_treino = st.session_state["y_treino"]
            x_validacao = st.session_state["x_validacao"]
            y_validacao = st.session_state["y_validacao"]

            # Construir rede
            modelo = Sequential()
            modelo.add(Flatten(input_shape=(28, 28)))
            modelo.add(Dense(128, activation='relu'))
            modelo.add(Dense(128, activation='relu'))
            modelo.add(Dense(10, activation='softmax'))

            modelo.compile(optimizer=optimizer_choice,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

            st.write("**Resumo do Modelo**:")
            st.text(modelo.summary())

            # Barra de progresso
            progress_bar = st.progress(0)
            callback_pb = ProgressBarCallback(progress_bar, total_epochs=epochs)

            # Mostrar spinner (mensagem de "carregando")
            with st.spinner("Treinando o modelo, aguarde..."):
                history = modelo.fit(
                    x_treino, y_treino,
                    epochs=epochs,
                    validation_data=(x_validacao, y_validacao),
                    verbose=0,  # deixamos 0 porque estamos usando callback
                    callbacks=[callback_pb]
                )

            st.session_state["modelo"] = modelo
            st.session_state["history"] = history.history

            st.success("Treinamento concluído!")
            df_hist = pd.DataFrame(history.history)
            st.write("**Curvas de Loss (erro) e Acurácia**:")
            st.line_chart(df_hist[["loss", "val_loss"]])
            st.line_chart(df_hist[["accuracy", "val_accuracy"]])
        else:
            st.error("Primeiro faça a divisão e normalização dos dados.")
else:
    st.warning("Carregue o MNIST para prosseguir.")

# -------------------------------------------------------------------------------------------------
# Seção 3: Avaliar no Conjunto de Teste
# -------------------------------------------------------------------------------------------------
st.subheader("3) Avaliar Modelo no Conjunto de Teste")
st.write("""
Depois de treinar, vamos ver como ele se sai em dados novos (o conjunto de teste).
Se a acurácia for > 90% ou 95%, já está ótimo para um exemplo simples!
""")

if st.session_state.get("mnist_loaded") and st.session_state.get("modelo"):
    if st.button("Avaliar no Teste"):
        x_teste = st.session_state["x_teste"] / 255.0
        y_teste = st.session_state["y_teste"]
        loss_test, acc_test = st.session_state["modelo"].evaluate(x_teste, y_teste)
        st.write(f"**Loss (teste)**: {loss_test:.4f}")
        st.write(f"**Acurácia (teste)**: {acc_test:.4%}")
else:
    st.info("É preciso ter o MNIST carregado e um modelo treinado/carregado para avaliar.")

# -------------------------------------------------------------------------------------------------
# Seção 4: Salvar / Carregar Modelo
# -------------------------------------------------------------------------------------------------
st.subheader("4) Salvar / Carregar o Modelo (Somente Pesos)")
st.write("""
Se quiser **guardar** o que a rede aprendeu (os “pesos”), basta salvar em um arquivo.  
Depois, você pode **carregar** esses pesos para não precisar treinar de novo.

**Atenção**: o Keras pode exigir que o nome termine em `.weights.h5` ao usar `save_weights()`.
""")

nome_arquivo = st.text_input("Nome do arquivo de modelo (ex: modelo_mnist_canvas.weights.h5)",
                             "modelo_mnist_canvas.weights.h5")

col_save, col_load = st.columns(2)

with col_save:
    if st.button("Salvar Pesos do Modelo"):
        if st.session_state.get("modelo"):
            if not nome_arquivo.endswith(".weights.h5"):
                st.error("O nome do arquivo deve terminar em `.weights.h5`.")
            else:
                st.session_state["modelo"].save_weights(nome_arquivo)
                st.success(f"Pesos salvos como {nome_arquivo}")
        else:
            st.error("Nenhum modelo para salvar.")

with col_load:
    if st.button("Carregar Pesos do Modelo"):
        if not nome_arquivo.endswith(".weights.h5"):
            st.error("O arquivo de pesos deve terminar em `.weights.h5`.")
        else:
            try:
                tmp_model = Sequential()
                tmp_model.add(Flatten(input_shape=(28, 28)))
                tmp_model.add(Dense(128, activation='relu'))
                tmp_model.add(Dense(128, activation='relu'))
                tmp_model.add(Dense(10, activation='softmax'))
                tmp_model.load_weights(nome_arquivo)

                tmp_model.compile(optimizer='adam',
                                  loss='sparse_categorical_crossentropy',
                                  metrics=['accuracy'])

                st.session_state["modelo"] = tmp_model
                st.success(f"Pesos carregados de {nome_arquivo}")
            except Exception as e:
                st.error(f"Erro ao carregar: {e}")


# -------------------------------------------------------------------------------------------------
# Seção 5: Predição no Canvas
# -------------------------------------------------------------------------------------------------
st.subheader("5) Desenhe um Dígito no Canvas e Peça para o Modelo Adivinhar")
st.write("""
Desenhe um dígito (0 a 9) no quadrado abaixo e clique em "Predizer Dígito".  
O app converte seu desenho para 28x28, inverte as cores (para combinar com o MNIST) e pergunta ao modelo: “O que é isso?”.

Se quiser apagar e tentar outro desenho, clique em "Resetar Canvas".
""")

if st.button("Resetar Canvas"):
    st.session_state["canvas_digit"] = None
    st.session_state["canvas_reset"] = True  # Flag para resetar o canvas
else:
    st.session_state["canvas_reset"] = False

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_color="black",
    stroke_width=10,
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas_digit",
    update_streamlit=st.session_state["canvas_reset"]  # Atualiza com base na flag
)


if st.button("Predizer Dígito"):
    if canvas_result and canvas_result.image_data is not None:
        if not st.session_state.get("modelo"):
            st.error("Não há modelo treinado/carregado para fazer a predição.")
        else:
            # Converte o canvas para imagem (utilizando OpenCV para processar)
            img = canvas_result.image_data.astype(np.uint8)  # Dados da imagem como uint8
            img = cv2.resize(img, (28, 28))  # Redimensiona para 28x28
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  # Converte para escala de cinza

            # Ajusta o contraste com threshold (dígito claro, fundo escuro)
            _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

            # Normaliza para [0, 1]
            img = img / 255.0

            # Visualiza o pré-processamento (opcional para depuração)
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            st.pyplot(fig)

            # Ajusta o formato para o modelo
            img = img.reshape(1, 28, 28)

            # Faz a predição
            preds = st.session_state["modelo"].predict(img)
            pred_digit = np.argmax(preds[0])

            st.write(f"**Dígito previsto**: {pred_digit}")
            st.write("**Probabilidades para cada dígito:**")
            st.bar_chart(preds[0])
    else:
        st.warning("Desenhe algo no canvas antes de clicar em 'Predizer Dígito'.")



st.markdown("---")

st.markdown("""
**Obrigado por utilizar esta demonstração!**  
Criado por [Vinicius Farineli Freire](https://www.linkedin.com/in/viniciusfarinelifreire/).
Aproveite e compartilhe se gostou!
""")
