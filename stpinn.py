import streamlit as st
import importlib
import torch
import numpy as np
from pinn_utils import pinn
import matplotlib.pyplot as plt
import time
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
        y_trial, checkpoints = pinn.solve(F, x0, [y0], NN, x_train, epochs=epochs, val_size = 0.1, lr=lr, return_checkpoints=True)
        plot_placeholder = st.empty()
        for checkpoint in checkpoints:
            fig, ax = plt.subplots()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid()
    
            # Prediction
            y_pred = checkpoint[1](x_train)
            ax.plot(x_train.detach().numpy(), y_pred.detach().numpy(), label="Prediction")
    
            # True solution, if available
            if true_sol is not None:
                y_true = true_sol(x_train.detach().numpy())
                ax.plot(x_train.detach().numpy(), y_true, label="True Solution", linestyle="--")
    
            ax.set_title(f"Solution to {ode_choice}, y({x0}) = {y0} \n Epoch {checkpoint[0]}")
            ax.legend()
    
            # Update the same plot each iteration
            plot_placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.025)
        st.markdown("### Final PINN Solution")
        fig_final, ax_final = plt.subplots()
        final_pred = y_trial(x_train)
        ax_final.plot(x_train.detach().numpy(), final_pred.detach().numpy(), label="Final Prediction")

        if true_sol is not None:
            y_true = true_sol(x_train.detach().numpy())
            ax_final.plot(x_train.detach().numpy(), y_true, linestyle="--", label="True Solution")

        ax_final.set_title("Final PINN Solution")
        ax_final.set_xlabel("x")
        ax_final.set_ylabel("y")
        ax_final.grid()
        ax_final.legend()
        st.pyplot(fig_final)
        plt.close(fig_final)