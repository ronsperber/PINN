import streamlit as st
import importlib
import torch
import numpy as np
from pinn_utils import pinn
import matplotlib.pyplot as plt
importlib.reload(pinn)

st.title("PINN ODE Solver")

with st.sidebar:
    st.header("Settings")
    ode_choice = st.selectbox("Choose ODE", ["y' = y", "y' = -y", "y' = y^2 - x"])
    x0 = st.number_input("x0", value=0.0)
    y0 = st.number_input("y(x0)", value=1.0)
    x_start = st.number_input("x start", value=0.0)
    x_end = st.number_input("x end", value=1.0)
    n_points = st.number_input("Number of points", value=100, step=10)
    epochs = st.number_input("Epochs", value=500, step=100)
    lr = st.number_input("Learning rate", value=1e-3, format="%.5f")
    num_hidden_layers = st.number_input("Hidden layers", value=2, step=1)
    layer_width = st.number_input("Layer width", value=64, step=1)

if st.button("Solve"):
    x_train = torch.linspace(x_start, x_end, n_points).reshape(-1, 1).requires_grad_(True)
    # Map choice to function
    if ode_choice == "y' = y":
        F = lambda x, y, dy: dy - y
        true_sol = lambda x : y0/np.exp(x0) * np.exp(x)
    elif ode_choice == "y' = -y":
        F = lambda x, y, dy: dy + y
        true_sol = lambda x : y0/np.exp(-x0) * np.exp(-x)
    elif ode_choice == "y' = y^2 - x":
        F = lambda x, y, dy: dy - (y**2 - x)
        true_sol = None

    NN = pinn.PINN(num_hidden_layers=num_hidden_layers, layer_width=layer_width)
    with st.spinner("Solving..."):
        y_trial = pinn.solve(F, x0, [y0], NN, x_train, epochs=epochs, val_size = 0.1, lr=lr)
    y_pred = y_trial(x_train)
    x_np = x_train.detach().numpy()
    y_np = y_pred.detach().numpy()
    fig, ax = plt.subplots()
    ax.plot(x_np, y_np, label="Prediction")
    if true_sol is not None:
        y_true = true_sol(x_np)
        ax.plot(x_np, y_true, label = "True Solution", linestyle = "--")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

