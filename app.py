import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.datasets import mnist

# Controle de páginas
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "main"

# Função para alternar páginas
def go_to(page):
    st.session_state["current_page"] = page

# Página principal
if st.session_state["current_page"] == "main":
    st.title("Reconhecimento de Dígitos (MNIST)")
    st.write("Bem-vindo ao aplicativo de reconhecimento de dígitos!")

    st.markdown("### 1. Carregar e Explorar o Dataset MNIST")
    if st.button("Ir para Carregar MNIST"):
        go_to("load_data")

    st.markdown("### 2. Treinar o Modelo")
    if st.button("Ir para Treinar Modelo"):
        go_to("train_model")

    st.markdown("### 3. Testar o Modelo no Canvas")
    if st.button("Ir para o Canvas"):
        go_to("canvas")

# Página para carregar e explorar o dataset
elif st.session_state["current_page"] == "load_data":
    st.title("Carregar e Explorar o Dataset MNIST")
    if "mnist_loaded" not in st.session_state:
        st.session_state["mnist_loaded"] = False

    if st.button("Carregar MNIST"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        st.session_state["x_train"] = x_train / 255.0  # Normalizar
        st.session_state["y_train"] = y_train
        st.session_state["x_test"] = x_test / 255.0
        st.session_state["y_test"] = y_test
        st.session_state["mnist_loaded"] = True
        st.success("Dataset MNIST carregado com sucesso!")

    if st.session_state["mnist_loaded"]:
        st.write("Exemplo de imagens no dataset:")
        num_images = st.slider("Quantas imagens mostrar?", 1, 10, 5)
        fig, axs = plt.subplots(1, num_images, figsize=(10, 2))
        for i in range(num_images):
            axs[i].imshow(st.session_state["x_train"][i], cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"Dígito: {st.session_state['y_train'][i]}")
        st.pyplot(fig)
    else:
        st.info("Clique no botão acima para carregar o dataset.")

    if st.button("Voltar"):
        go_to("main")

# Página para treinar o modelo
elif st.session_state["current_page"] == "train_model":
    st.title("Treinar Modelo de Rede Neural")
    if "modelo" not in st.session_state:
        st.session_state["modelo"] = None

    if st.session_state.get("mnist_loaded"):
        st.write("Configuração de Treinamento:")
        epochs = st.slider("Número de Épocas", 1, 20, 5)
        optimizer_choice = st.selectbox("Otimizador", ["adam", "sgd"])

        if st.button("Treinar Modelo"):
            x_train = st.session_state["x_train"]
            y_train = st.session_state["y_train"]
            modelo = Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(10, activation='softmax')
            ])
            modelo.compile(optimizer=optimizer_choice, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            history = modelo.fit(x_train, y_train, epochs=epochs, validation_split=0.1, verbose=0)
            st.session_state["modelo"] = modelo
            st.success("Modelo treinado com sucesso!")

            st.write("Curvas de aprendizado:")
            history_df = pd.DataFrame(history.history)
            st.line_chart(history_df[["loss", "val_loss"]])
            st.line_chart(history_df[["accuracy", "val_accuracy"]])
    else:
        st.warning("Você precisa carregar o MNIST primeiro.")

    if st.button("Voltar"):
        go_to("main")

# Página do Canvas
elif st.session_state["current_page"] == "canvas":
    st.title("Testar Modelo no Canvas")
    if "canvas_image_data" not in st.session_state:
        st.session_state["canvas_image_data"] = None

    if st.button("Resetar Canvas"):
        st.session_state["canvas_image_data"] = None

    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_color="#FFFFFF",
        stroke_width=20,
        background_color="#000000",
        width=256,
        height=256,
        drawing_mode="freedraw",
        key="canvas_digit",
        initial_drawing=st.session_state["canvas_image_data"]
    )

    if canvas_result and canvas_result.image_data is not None:
        st.session_state["canvas_image_data"] = canvas_result.image_data

    if st.button("Predizer Dígito"):
        if st.session_state.get("modelo") and st.session_state["canvas_image_data"] is not None:
            img = st.session_state["canvas_image_data"].astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            img = img / 255.0
            img = img.reshape(1, 28, 28)
            preds = st.session_state["modelo"].predict(img)
            pred_digit = np.argmax(preds[0])
            st.write(f"**Dígito previsto:** {pred_digit}")

            st.bar_chart(preds[0])
        else:
            st.error("Modelo não treinado ou canvas vazio.")

    if st.button("Voltar"):
        go_to("main")
