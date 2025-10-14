import streamlit as st
import importlib
import torch
import numpy as np
from pinn_utils import pinn
from pinn_utils.de_sols import (
    bernoulli_sol,
    cauchy_euler_sol, 
    exp_sol, 
    logistic_sol, 
    linear_nonhomogeneous_sol,
    linear_homogeneous_sol
)
import plotly.graph_objects as go
import time
importlib.reload(pinn)

st.title("PINN ODE Solver")

with st.sidebar:
    st.header("Settings")
    ode_choice = st.selectbox(
        "Choose ODE",
        [
            "y' = k y",
            "y' = k y (1 - y)",
            "y' = k y + sin(x)",
            "y' = k y\u00B2",
            "y'' + by' + cy = 0",
            "x\u00B2 y'' + b x y' + c y = 0"
        ]
    )
    ode_orders = {
        "y' = k y":1, 
        "y' = k y (1 - y)":1, 
        "y' = k y + sin(x)":1,
        "y' = k y\u00B2":1,
        "y'' + by' + cy = 0":2,
        "x\u00B2 y'' + b x y' + c y = 0":2
    }
    if ode_choice in ["y' = k y", "y' = k y (1 - y)", "y' = k y + sin(x)", "y' = k y\u00B2"]:
        k = st.number_input("k", value=1.0)
    if ode_choice == "x\u00B2 y'' + b x y' + c y = 0":
        x0 = st.number_input("x0 (must be > 0)", min_value=1e-6, value=1.0, key="x0_pos")
    else:
        x0 = st.number_input("x0", value=0.0, key="x0_any")
    y0 = st.number_input("y(x0)", value=1.0)
    if ode_orders[ode_choice] == 2:
        yprime0 = st.sidebar.number_input("y'(x0)", value=0.0)
    else:
        yprime0 = None
    if ode_choice in ("y'' + by' + cy = 0", "x\u00B2 y'' + b x y' + c y = 0") :
        b = st.sidebar.number_input("b", value=2.0)
        c = st.sidebar.number_input("c", value=1.0)
    if ode_choice == "x\u00B2 y'' + b x y' + c y = 0":
        x_start = st.number_input("x start (must be > 0)", min_value= min(x0, 1e-6), value=x0, key="x_start_positive")
        x_end = st.number_input("x end", min_value = x_start , value=x_start + 5.0, key ="x_end_pos")
    else:
        x_start = st.number_input("x start", value=x0, min_value = x0 - 5.0, key="x_start_default")
        x_end = st.number_input("x end", value=x_start + 2.0, min_value = x_start, key="x_end_default")
    with st.expander("Neural Network Parameters", expanded=False):
        st.caption("Tweak only if the solver struggles or you want to experiment.")
        n_points = st.number_input("Number of points in interval", value=100, step=10)
        epochs = st.number_input("Epochs", value=500, step=100)
        lr = st.number_input("Learning rate", value=1e-3, format="%.5f")
        num_hidden_layers = st.number_input("Hidden layers", value=2, step=1)
        layer_width = st.number_input("Layer width", value=64, step=1)
        activation_options = st.selectbox("Activation function", ["Softplus", "Tanh", "ReLU", "Swish"], index=0)
    activation_dict = {
        "Softplus": torch.nn.Softplus(),
        "Tanh": torch.nn.Tanh(),
        "ReLU": torch.nn.ReLU(),
        "Swish": lambda x: x * torch.sigmoid(x)
    }
activation = activation_dict[activation_options]
if ode_choice == "y' = k y":
    ode = f"dy/dx = {k} y"
elif ode_choice == "y' = k y (1 - y)":
    ode = f"dy/dx = {k} y (1 - y)"
elif ode_choice == "y' = k y + sin(x)":
    ode = f"dy/dx = {k} y + sin(x)"
elif ode_choice == "y' = k y\u00B2":
    ode = f"dy/dx = {k} y²"
elif ode_choice == "y'' + by' + cy = 0":
    ode = f"d²y/dx² + {b} dy/dx + {c} y = 0"
elif ode_choice == "x\u00B2 y'' + b x y' + c y = 0":
    ode = f"x² d²y/dx² + {b} x dy/dx + {c} y = 0"
# Detect sidebar parameter changes and clear previous frames if any parameter changed
current_params = dict(ode_choice=ode_choice, x0=float(x0), y0=float(y0), x_start=float(x_start), x_end=float(x_end), n_points=int(n_points), epochs=int(epochs), lr=float(lr), num_hidden_layers=int(num_hidden_layers), layer_width=int(layer_width))
if 'last_params' not in st.session_state:
    st.session_state['last_params'] = current_params
elif st.session_state['last_params'] != current_params:
    # user changed parameters — clear any previously computed frames so chart doesn't persist
    st.session_state.pop('frames', None)
    st.session_state.pop('x_np', None)
    st.session_state.pop('png_bytes', None)
    st.session_state.pop('gif_bytes', None)
    st.session_state['last_params'] = current_params

import numpy as np



col1, col2 = st.columns([1,1])
with col1:
    solve_clicked = st.button("Solve")
with col2:
    reset_clicked = st.button("Reset")

if reset_clicked:
    # Clear previously computed frames and related UI state
    for k in ['frames', 'x_np', 'current_idx', 'slider_idx', 'last_slider_value', 'playing']:
        st.session_state.pop(k, None)
    st.session_state.pop('png_bytes', None)
    st.session_state.pop('gif_bytes', None)

if solve_clicked:
    x_train = torch.linspace(x_start, x_end, n_points).reshape(-1, 1).requires_grad_(True)

    # Map choice to function
    if ode_choice == "y' = k y":
        F = lambda x, y, dy: dy - k * y
        true_sol = exp_sol(k=k, x0=x0, y0=y0)  # exponential growth
    elif ode_choice == "y' = k y (1 - y)":
        F = lambda x, y, dy: dy - k *y * (1 - y)
        if y0 in (0,1):
            true_sol = lambda x: y0 * np.ones_like(x) # constant solution
        else:
            true_sol = logistic_sol(k=k, x0=x0, y0=y0)  # logistic solution
    elif ode_choice == "y' = k y + sin(x)":
        F = lambda x, y, dy: dy - k * y - torch.sin(x)
        true_sol = linear_nonhomogeneous_sol(k=k, x0=x0, y0=y0)  
    elif ode_choice == "y' = k y\u00B2":
        F = lambda x, y, dy: dy - k * y**2
        true_sol = bernoulli_sol(k=k, x0=x0, y0=y0)
    elif ode_choice == "y'' + by' + cy = 0":
        F = lambda x, y, dy, ddy: ddy + b * dy +  c * y
        true_sol = linear_homogeneous_sol(x0=x0, y0=y0, yprime0=yprime0, b=b, c=c)
    elif ode_choice == "x\u00B2 y'' + b x y' + c y = 0":
        F = lambda x, y, dy, ddy: x**2 * ddy + b * x * dy + c * y
        # Characteristic equation: r^2 + (b-1) r + c = 0
        true_sol = cauchy_euler_sol(x0=x0, y0=y0, yprime0=yprime0, b=b, c=c)


    NN = pinn.PINN(
        num_hidden_layers=num_hidden_layers,
        layer_width=layer_width,
        input_activation=activation,
        hidden_activation=activation
        )
    if ode_orders[ode_choice] == 1:
        ics = [y0]
    else:
        ics = [y0, yprime0]
    with st.spinner("Solving..."):
        y_trial, checkpoints = pinn.solve(F,
                                          x0,
                                          ics,
                                          NN,
                                          x_train,
                                          epochs=epochs,
                                          val_size=0.1, 
                                          lr=lr,
                                          return_checkpoints=True)

        # Build frames (prediction, optional true) and store in session_state
        x_np = x_train.detach().numpy()
        frames = []
        for checkpoint in checkpoints + [("final", y_trial)]:
            y_pred = checkpoint[1](x_train).detach().numpy()
            y_true = true_sol(x_np) if true_sol is not None else None
            if isinstance(checkpoint[0], int):
                title = f"Solution to {ode_choice}, y({x0}) = {y0}\nEpoch {checkpoint[0]}"
                epoch_val = int(checkpoint[0])
            else:
                if ode_orders[ode_choice] == 1:
                    title = f"Final Solution to {ode}, y({x0}) = {y0}"
                else:
                    title = f"Final Solution to {ode}, y({x0}) = {y0}, y'({x0}) = {yprime0}"
                epoch_val = "final"
            frames.append({"y_pred": y_pred, "y_true": y_true, "title": title, "epoch": epoch_val})

    st.session_state['frames'] = frames
    st.session_state['x_np'] = x_np
    # clear previous exports when new solution is computed
    st.session_state.pop('png_bytes', None)
    st.session_state.pop('gif_bytes', None)


if 'frames' in st.session_state:
    frames = st.session_state['frames']
    x_np = st.session_state['x_np']
    n_frames = len(frames)

    # Prepare Plotly animated figure (client-side animation)
    x = x_np.flatten()
    has_true = any(fr['y_true'] is not None for fr in frames)

    # Initialize figure using the final frame so users see the final solution by default
    final_idx = n_frames - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=frames[final_idx]['y_pred'].flatten(), mode='lines', name='Prediction'))
    if has_true:
        fig.add_trace(go.Scatter(x=x, y=frames[final_idx]['y_true'].flatten(), mode='lines', name='True Solution', line=dict(dash='dash')))

    plotly_frames = []
    for i, fr in enumerate(frames):
        data = [go.Scatter(x=x, y=fr['y_pred'].flatten())]
        annotations = []
        if fr['y_true'] is not None:
            data.append(go.Scatter(x=x, y=fr['y_true'].flatten()))
            mse = np.mean((fr['y_pred'].flatten() - fr['y_true'].flatten())**2)
            annotations.append(dict(
                xref='paper', yref='paper', x=0.95, y=0.95,  # top-right corner
                text=f"MSE: {mse:.6f}",
                showarrow=False,
                font=dict(size=12, color="black")
            ))

        plotly_frames.append(go.Frame(data=data, name=str(i), layout=go.Layout(title=fr['title'], annotations=annotations)))

    duration_ms = max(1, int(0.2 * 1000))
    # Use epoch labels on the slider steps (show 'Epoch N' or 'Final')
    steps = []
    for i, fr in enumerate(frames):
        epoch = fr.get('epoch')
        if isinstance(epoch, int):
            label = f"Epoch {epoch}"
        else:
            label = "Final"
        steps.append(dict(method='animate', args=[[str(i)], dict(mode='immediate', frame=dict(duration=duration_ms, redraw=True), transition=dict(duration=0))], label=label))
    sliders = [dict(active=final_idx, pad={'t': 50}, steps=steps, currentvalue={'prefix': 'Frame: '})]
    updatemenus = [dict(type='buttons', showactive=False, y=0, x=1.05, xanchor='right', yanchor='top', buttons=[
    # Ensure Play actually animates through frames from the start
    dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=duration_ms, redraw=True), transition=dict(duration=0), fromcurrent=False, mode='immediate')]),
        dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
    ])]

    fig.frames = plotly_frames
    fig.update_layout(updatemenus=updatemenus, sliders=sliders, title=frames[final_idx]['title'])

    st.plotly_chart(fig, use_container_width=True)

    # Export controls: Prepare and Download flow to avoid transient UI issues
    if 'png_bytes' not in st.session_state:
        st.session_state['png_bytes'] = None
    if 'gif_bytes' not in st.session_state:
        st.session_state['gif_bytes'] = None

    col_png, col_gif = st.columns([1, 1])
    with col_png:
        if st.button("Prepare final PNG"):
            try:
                png_bytes = fig.to_image(format='png')
                st.session_state['png_bytes'] = png_bytes
                st.success("PNG prepared — click Download PNG below.")
            except Exception as e:
                st.error(f"PNG export failed: {e}. Install the 'kaleido' package to enable Plotly PNG export.")

        if st.session_state.get('png_bytes') is not None:
            st.download_button("Download PNG", data=st.session_state['png_bytes'], file_name="pinn_final.png", mime="image/png")

    with col_gif:
        if st.button("Prepare GIF"):
            try:
                import io
                from PIL import Image

                imgs = []
                frames_count = len(plotly_frames)
                if frames_count == 0:
                    st.error("No frames available to build GIF.")
                else:
                    progress = st.progress(0)
                    est_text = st.empty()
                    with st.spinner("Preparing GIF — rendering frames..."):
                        # measure frames and provide a running "time left" estimate
                        sample_times = []
                        for idx, fr in enumerate(plotly_frames):
                            start = time.time()
                            temp_fig = go.Figure(data=fr.data, layout=fr.layout)
                            png = temp_fig.to_image(format='png')
                            img = Image.open(io.BytesIO(png)).convert('RGBA')
                            imgs.append(img)
                            elapsed = time.time() - start
                            sample_times.append(elapsed)
                            avg = sum(sample_times) / len(sample_times)
                            frames_left = frames_count - (idx + 1)
                            est_left = avg * frames_left
                            est_text.text(f"Estimated time left: {est_left:.1f}s")
                            progress.progress(int((idx + 1) / frames_count * 100))
                        est_text.empty()

                    bio = io.BytesIO()
                    imgs[0].save(bio, format='GIF', save_all=True, append_images=imgs[1:], duration=200, loop=0)
                    bio.seek(0)
                    st.session_state['gif_bytes'] = bio.getvalue()
                    st.success("GIF prepared — click Download GIF below.")
            except Exception as e:
                st.error(f"GIF export failed: {e}. Ensure 'kaleido' and 'Pillow' are installed for export.")

        if st.session_state.get('gif_bytes') is not None:
            st.download_button("Download GIF", data=st.session_state['gif_bytes'], file_name='pinn_evolution.gif', mime='image/gif')

    # Show MSE for final frame if true solution is available
    final_frame = frames[-1]
    if final_frame.get('y_true') is not None:
        mse = ((final_frame['y_true'] - final_frame['y_pred'])**2).mean()
        st.write(f"MSE for PINN solution (final frame): {mse:.8f}")