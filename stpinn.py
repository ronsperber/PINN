import streamlit as st
import importlib
import pandas as pd
import torch
import numpy as np
import plotly.graph_objects as go
import time
from pathlib import Path
import os
from pinn_utils import pinn
from pinn_utils.ode_meta import ODES
def read_markdown_file(file_path):
    """Reads the content of a Markdown file."""
    return Path(file_path).read_text()
# read the markdown with mathematical background
math_md = read_markdown_file("PINN_math.md")
importlib.reload(pinn)
st.title("Solving ODEs using a PINN (Physics Informed Neural Network)")
st.write("To see a differential equation solved using a PINN, select an equation type on the left, adjust any desired parameters, and press solve.")
with st.expander("Expand to see the mathematics behind this method.", expanded=False):
    st.markdown(math_md)
# Sidebar inputs driven by ODES metadata
ode_choice = st.sidebar.selectbox("Choose ODE", list(ODES.keys()))


on_streamlit_cloud = "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION" in os.environ
if on_streamlit_cloud:
    make_gif = st.sidebar.checkbox(
        "Generate animation GIF (disabled on Streamlit Cloud)",
        value=False,
        disabled=True,
        help="Creating a GIF requires Kaleido to use Chrome, which isn't supported on Streamlit Cloud."
    )
else:
    make_gif = st.sidebar.checkbox(
    "Create a GIF from the frames produced (may be slow)",
    value=False,
    help="Generating a GIF requires rendering each frame, which can take some time"
    )
meta = ODES[ode_choice]
if meta.get("is_system", False):
    # System inputs
    A11 = st.sidebar.number_input("A₁₁", value=0.0)
    A12 = st.sidebar.number_input("A₂₁", value=1.0)
    A21 = st.sidebar.number_input("A₂₁", value=-1.0)
    A22 = st.sidebar.number_input("A₂₂", value=0.0)
    x0 = st.sidebar.number_input("t₀", value=0.0)
    y1_0 = st.sidebar.number_input("y₁(t₀)", value=1.0)
    y2_0 = st.sidebar.number_input("y₂(t₀)", value=0.0)
    
    y0 = torch.tensor([y1_0, y2_0])
    x_start =st.sidebar.number_input("t_start", value=0.0)
    x_end = st.sidebar.number_input("t_end", value = x_start + 2.0)
    # non-applicable scalar ODE params
    k = b = c = yprime0 = None
else:
    # Small tolerance to avoid float-boundary validation edge-cases in Streamlit inputs
    EPS = 1e-9
    # non-applicable system ODE params
    A11 = A12 = A21 = A22 = None
    # Basic ICs and parameters (in sidebar)
    if meta.get('x0_positive'):
        # enforce positive x0 for ODEs that require it (e.g., Cauchy-Euler)
        x0 = st.sidebar.number_input("x0", value=1.0, min_value=1e-6)
    else:
        x0 = st.sidebar.number_input("x0", value=0.0)
    if meta.get('needs_k'):
        k = st.sidebar.number_input("k", value=1.0)
    else:
        k = None
    y0 = st.sidebar.number_input("y(x0)", value=1.0)
    if meta.get('order', 1) == 2:
        yprime0 = st.sidebar.number_input("y'(x0)", value=0.0)
    else:
        yprime0 = None
    if meta.get('needs_b_c'):
        b = st.sidebar.number_input("b", value=2.0)
        c = st.sidebar.number_input("c", value=1.0)
    else:
        b = None
        c = None
    if meta.get('x0_positive'):
        # allow a tiny tolerance so users can type values like 0.50 without floating-point validation errors
        x_start_min = max(1e-6, x0 - EPS)
        x_start = st.sidebar.number_input("x start (must be > 0)", min_value=x_start_min, value=x0, key="x_start_positive")
        x_end = st.sidebar.number_input("x end", min_value=max(x_start - EPS, 1e-6), value=x_start + 5.0, key="x_end_pos")
    else:
        x_start = st.sidebar.number_input("x start", value=x0, min_value=x0 - 5.0 - EPS, key="x_start_default")
        x_end = st.sidebar.number_input("x end", value=x_start + 2.0, min_value=x_start - EPS, key="x_end_default")
# add optional parameters for the Neural network to potentially improve training
with st.sidebar.expander("Neural Network Parameters", expanded=False):
    st.caption("Tweak only if the solver struggles or you want to experiment. Choosing very large values for epochs, number of hidden layers, and/or layer width could make the solver slow.")
    n_points = st.number_input("Number of points in interval", min_value= 10, value=100, step=10)
    epochs = st.number_input("Epochs", value=500, step=100,min_value=10,max_value=10000)
    lr = st.number_input("Learning rate", value=1e-3, format="%.5f")
    num_hidden_layers = st.number_input("Hidden layers", value=2, step=1)
    layer_width = st.number_input("Layer width", value=64, step=1,min_value=2,max_value=1024)
    activation_options = st.selectbox("Activation function",
                                      [ "Tanh", "Sine", "Swish", "Softplus"],
                                      index=0)

activation_dict = {
    "Tanh": torch.nn.Tanh(),
    "Sine" : torch.sin,
    "Swish": lambda x: x * torch.sigmoid(x),
    "Softplus": torch.nn.Softplus()
}
# parameters to be passed to factories (F_factory and true_sol_factory)
params = {
    "x0" : x0,
    "y0" : y0,
    "yprime0": yprime0,
    "k" : k,
    "b" : b,
    "c" : c,
    "A11": A11,
    "A12": A12,
    "A21": A21,
    "A22": A22,
}

activation = activation_dict[activation_options]

# Build a human-readable ODE string for titles
ode = meta.get('ode_str', lambda **kw: ode_choice)(**params)
# function to safely store vectors in params for comparison

def to_serializable(val):
    """
    convert tensors to lists for storing as session parameters
    used so we can compare current to stored parameters 
    """
    if torch.is_tensor(val):
        return val.detach().cpu().numpy().tolist()  # convert to list
    elif isinstance(val, (list, tuple)):
        # if we have a list or tuple of parameters, make sure we convert each element
        return [to_serializable(v) for v in val]
    else:
        return val

# Detect sidebar parameter changes and clear previous frames if any parameter changed
# current_params holds all sidebar parameters
current_params = dict(
    ode_choice=ode_choice,
    k=k,
    b=b,
    c=c,
    A11=A11,
    A12=A12,
    A21=A21,
    A22=A22,
    x0=x0,
    y0=to_serializable(y0),
    yprime0=to_serializable(yprime0),
    x_start=x_start,
    x_end=x_end,
    n_points=int(n_points),
    epochs=int(epochs),
    lr=float(lr),
    num_hidden_layers=int(num_hidden_layers),
    layer_width=int(layer_width),
    activation=activation_options
    )
if 'last_params' not in st.session_state:
    st.session_state['last_params'] = current_params
elif st.session_state['last_params'] != current_params:
    print("=== Parameter differences detected ===")
    for key in current_params:
        old_val = st.session_state['last_params'].get(key)
        new_val = current_params.get(key)
        if old_val != new_val:
            print(f"{key}: old={old_val} | new={new_val}")
    print("====================================")
    # user changed parameters — clear any previously computed frames so chart doesn't persist
    st.session_state.pop('frames', None)
    st.session_state.pop('x_np', None)
    st.session_state.pop('plotly_frames', None)
    st.session_state.pop('png_bytes', None)
    st.session_state.pop('gif_bytes', None)
    st.session_state.pop('fig', None)
    st.session_state['last_params'] = current_params 

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
    # Validate positive-domain requirements before running solver
    if meta.get('x0_positive', False) and x0 <= 0:
        st.sidebar.error("This ODE requires x0 > 0. Please set x0 to a positive value.")
    elif meta.get('x0_positive', False) and x_start <= 0:
        st.sidebar.error("This ODE requires the interval start to be > 0. Please set x start to a positive value.")
    else:
        # create the interval used to train the data
        x_train = torch.linspace(x_start, x_end, n_points).reshape(-1, 1).requires_grad_(True)
        # Build F and true solution using ODES metadata
        meta = ODES[ode_choice]
        # Create the residual function F using the factory. Factories accept k, b, c etc and ignore extras.
        F = meta['F_factory'](**params)
        # Build true solution if factory provided
        true_factory = meta.get('true_factory')
        if callable(true_factory):
            true_sol = None
            true_factory_err = None
            try:
                true_sol = true_factory(**params)
            except Exception as e:
                true_factory_err = e
                try:
                    true_sol = true_factory(x0=x0, y0=y0)
                    true_factory_err = None
                except Exception as e2:
                    true_factory_err = e2
                    true_sol = None
            # Report analytic-solution availability to the sidebar for debugging
            if true_sol is None:
                st.sidebar.warning("Analytic true solution unavailable for selected parameters.")
                if true_factory_err is not None:
                    st.sidebar.caption(f"True-factory error: {true_factory_err}")
            else:
                st.sidebar.success("Analytic true solution built.")
        else:
            true_sol = None
        # create the PINN to be used
        num_outputs = 1
        if meta.get("is_system", False):
            num_outputs = 2
        NN = pinn.PINN(
            num_hidden_layers=num_hidden_layers,
            layer_width=layer_width,
            input_activation=activation,
            hidden_activation=activation,
            num_outputs=num_outputs
            )
        # create list of ICs
        if meta.get('order', 1) == 1:
            ics = [y0]
        else:
            ics = [y0, yprime0]
        # solve the DE with a progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        # Record start time so we can estimate remaining time
        start_time = time.time()

        def _format_seconds(s: float) -> str:
            """
            Format seconds into H:MM:SS (or MM:SS) for display
            """
            if s is None or s != s or s < 0:
                return "-"
            s = int(round(s))
            h, rem = divmod(s, 3600)
            m, sec = divmod(rem, 60)
            if h:
                return f"{h}:{m:02d}:{sec:02d}"
            return f"{m:02d}:{sec:02d}"

        def _progress_callback(epoch, train_loss, val_loss):
            """
            callback function passed to solve to use
            for displaying progress and ETA
            """
            # epoch ranges from 1..epochs; clamp and compute fraction
            try:
                frac = min(max(epoch / max(1, int(epochs)), 0.0), 1.0)
            except Exception:
                frac = 0.0
            progress_bar.progress(int(frac * 100))
            # estimate remaining time based on avg seconds per epoch so far
            elapsed = time.time() - start_time
            eta = None
            try:
                if epoch > 0:
                    avg = elapsed / epoch
                    eta = avg * max(0, int(epochs) - epoch)
            except Exception:
                eta = None
            eta_str = _format_seconds(eta) if eta is not None else "-"
            progress_text.text(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6e}, val_loss: {val_loss:.6e} — ETA: {eta_str}")

        with st.spinner("Solving..."):
            checkpoint_every = min(max(1, epochs//25),20)
            y_trial, checkpoints = pinn.ode_solve(F=F,
                                                  a=x0,
                                                  ics=ics,
                                                  NN=NN,
                                                  return_checkpoints=True,
                                                  checkpoint_every=checkpoint_every,
                                                  X=x_train,
                                                  epochs=epochs,
                                                  val_size=0.1, 
                                                  lr=lr,
                                                  progress_callback=_progress_callback,
                                                  progress_every=10)

        # clear progress UI
        progress_bar.empty()
        progress_text.empty()
        x_np = x_train.detach().numpy()
        st.session_state["x_np"] = x_np.flatten()
        # Build frames (prediction, optional true) and store in session_state
        frames = []
        # helper to reconstruct a NN from checkpoint state (if provided)
        def _nn_from_checkpoint_fn(ck_fn):
            state = None
            try:
                defaults = ck_fn.__defaults__
                if defaults and len(defaults) > 0:
                    state = defaults[0]
            except Exception:
                state = None
            if state is None:
                return None
            # build a fresh NN with same architecture
            nn_copy = pinn.PINN(
                num_hidden_layers=num_hidden_layers, 
                layer_width=layer_width,
                input_activation=activation,
                hidden_activation=activation,
                num_outputs=num_outputs
            )
            nn_copy.load_state_dict(state)
            nn_copy.eval()
            return nn_copy

        # build frames from each checkpoint
        for checkpoint in checkpoints + [("final", y_trial)]:
            # checkpoints are pairs (epoch, intermediate solution)
            ck_fn = checkpoint[1] 
            nn_for_eval = _nn_from_checkpoint_fn(ck_fn)
            if nn_for_eval is None:
                # final frame: use the trained NN instance
                nn_for_eval = NN
            # build a differentiable trial function from this NN so we can compute derivatives/residual
            y_fn = pinn.get_y_trial(x0, ics, nn_for_eval)
            # ensure x_train requires grad for derivative computation
            x_for_eval = x_train.detach().clone().requires_grad_(True)
            # compute the ODE loss at each checkpoint
            y_torch = y_fn(x_for_eval)
            derivs = pinn.derivatives(y_torch, x_for_eval, len(ics))
            try:
                res = F(x_for_eval, *derivs)
                ode_loss = float(torch.mean(res**2).item())
            except Exception as e:
                ode_loss = None
            y_pred = y_torch.detach().numpy()
            # many analytic factories expect a 1-D numpy array; flatten to be safe
            y_true = true_sol(x_np.flatten()) if true_sol is not None else None
            if isinstance(checkpoint[0], int):
                title = f"Solution to {ode}, y({x0}) = {y0}\nEpoch {checkpoint[0]}"
                epoch_val = int(checkpoint[0])
            else:
                if meta.get('order', 1) == 1:
                    title = f"Final Solution to {ode}, y({x0}) = {y0}"
                else:
                    title = f"Final Solution to {ode}, y({x0}) = {y0}, y'({x0}) = {yprime0}"
                epoch_val = "final"
            frames.append({"y_pred": y_pred, "y_true": y_true, "title": title, "epoch": epoch_val, "ode_loss": ode_loss})

        st.session_state['frames'] = frames
        st.session_state['x_np'] = x_np
        # clear previous exports when new solution is computed
        st.session_state.pop('png_bytes', None)
        st.session_state.pop('gif_bytes', None)
if 'frames' in st.session_state:
    frames = st.session_state['frames']
    x_np = st.session_state['x_np']
    n_frames = len(frames)
    # Prepare Plotly animated figure 
    x = x_np.flatten()
    has_true = any(fr['y_true'] is not None for fr in frames)
    # Initialize figure using the final frame so users see the final solution by default
    final_idx = n_frames - 1
    fig = go.Figure()
    if meta.get("is_system", False):
        # phase-space trajectory plot
        y_pred_final = frames[final_idx]['y_pred']
        fig.add_trace(go.Scatter(
            x=y_pred_final[:, 0],
            y=y_pred_final[:, 1],
            mode='lines',
            name='Prediction'
        ))
        if has_true:
            y_true_final = frames[final_idx]['y_true']
            fig.add_trace(go.Scatter(
                x=y_true_final[:, 0],
                y=y_true_final[:, 1],
                mode='lines',
                name='True Solution',
                line=dict(dash='dash')
            ))
        fig.update_layout(
            xaxis_title="y₁(t)",
            yaxis_title="y₂(t)",
            title="Phase-space trajectory"
        )
    else:
        fig.add_trace(go.Scatter(x=x, y=frames[final_idx]['y_pred'].flatten(), mode='lines', name='Prediction'))
        if has_true:
            fig.add_trace(go.Scatter(x=x, y=frames[final_idx]['y_true'].flatten(), mode='lines', name='True Solution', line=dict(dash='dash')))

    plotly_frames = []
    for i, fr in enumerate(frames):
        if meta.get("is_system", False):
            data = [go.Scatter(x=fr['y_pred'][:,0].flatten(), y=fr['y_pred'][:,1].flatten(), mode='lines', name = 'Prediction')]
        else:
            data = [go.Scatter(x=x, y=fr['y_pred'].flatten(),mode='lines', name='Prediction')]

        if fr['y_true'] is not None:
            if meta.get("is_system", False):
                data.append(go.Scatter(x=fr['y_true'][:,0].flatten(), y=fr['y_true'][:,1].flatten(), mode='lines', name= 'True Solution', line=dict(dash='dash')))
            else:
                data.append(go.Scatter(x=x, y=fr['y_true'].flatten(), mode='lines', name='True Solution', line=dict(dash='dash')))
            mse = np.mean((fr['y_pred'].flatten() - fr['y_true'].flatten())**2)
            ann_text = f"MSE: {mse:.6f}"
            # include ODE residual when available
            ode_val = fr.get('ode_loss')
            if ode_val is not None:
                ann_text += f"<br>Residual: {ode_val:.3e}"
        else:
            # no true solution; still show residual if available
            ode_val = fr.get('ode_loss')
            if ode_val is not None:
                ann_text = f"Residual: {ode_val:.3e}"
            else:
                ann_text = None
        frame_title = fr['title']
        if ann_text is not None:
            frame_title += f"  | {ann_text.replace('<br>', ' | ')}"
        plotly_frames.append(go.Frame(data=data, name=str(i), layout=dict(title=frame_title)))
    n_frames = len(frames)
    duration_ms = max(10, min(200, int(5000 / n_frames)))
    st.session_state["plotly_frames"] = plotly_frames
    
    
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
    st.session_state['fig'] = fig
if st.session_state.get("fig") is not None:
    st.plotly_chart(st.session_state["fig"], use_container_width=True)
# @st.cache_data(show_spinner=False)
def frames_to_gif(plotly_frames, x, fps=10, is_system=False, max_line_length=40):
    import io
    import imageio.v2 as imageio
    import plotly.graph_objects as go, plotly.io as pio



    def wrap_text(text, max_len):
        """Insert <br> in text to wrap long titles."""
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_len:
                current_line = f"{current_line} {word}".strip()
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return "<br>".join(lines)

    images = []

    for fr in plotly_frames:
        fig = go.Figure()

        # Add each trace individually, preserving name and line style
        for trace in fr.data:
            fig.add_trace(go.Scatter(
                x=trace.x,
                y=trace.y,
                mode=trace.mode,
                name=trace.name or "",
                line=trace.line if hasattr(trace, 'line') else None
            ))

        # Set the frame's title with font size and wrapping
        if fr.layout.title and hasattr(fr.layout.title, 'text'):
            wrapped_title = wrap_text(fr.layout.title.text, max_line_length)
            fig.update_layout(title=dict(
                text=wrapped_title,
                x=0.5,  # center title
                xanchor='center',
                font=dict(size=16)
            ))

        # Fix x-axis range if not a system
        if not is_system:
            fig.update_layout(xaxis=dict(range=[x.min(), x.max()]))

        # Ensure enough space for title
        fig.update_layout(
            width=800,
            height=600,
            margin=dict(l=50, r=50, t=120, b=50),  # t=120 gives more space for wrapped title
        )

        # Convert figure to PNG bytes
        img_bytes = pio.to_image(fig, format="png")
        images.append(imageio.imread(io.BytesIO(img_bytes)))

    # Save images to GIF
    gif_bytes = io.BytesIO()
    imageio.mimsave(gif_bytes, images, format="GIF", fps=fps)
    gif_bytes.seek(0)
    return gif_bytes



# only generate GIF if user requested it
if make_gif and 'plotly_frames' in st.session_state:
    with st.spinner("Generating animation GIF..."):
        gif_bytes = frames_to_gif(
            st.session_state['plotly_frames'],
            st.session_state['x_np'],
            fps=10,
            is_system=meta.get("is_system", False)
        )
        st.session_state['gif_bytes'] = gif_bytes

    st.download_button(
        label="Download GIF",
        data=st.session_state['gif_bytes'],
        file_name="pinn_animation.gif",
        mime="image/gif"
    )
    # Show ODE residuals (mean squared residual) over frames if available using Plotly
    # user-controlled toggle to show/hide residuals
if "frames" in st.session_state:
    frames = st.session_state["frames"]
    ode_losses = [fr.get('ode_loss') for fr in frames]
    with st.expander("Show ODE residuals", expanded = False):
        # convert None -> nan for plotting
        yvals = [pl if pl is not None else float('nan') for pl in ode_losses]
        res_fig = go.Figure()
        res_fig.add_trace(go.Scatter(x=list(range(1, len(yvals) + 1)), y=yvals, mode='lines+markers', name='ODE residual'))
        res_fig.update_layout(title='ODE residual (per-frame)', xaxis_title='Checkpoint', yaxis_title='Mean ODE residual', template='plotly_white')
        st.plotly_chart(res_fig, use_container_width=True)
        # show final numeric value if available
        final_vals = [v for v in ode_losses if v is not None]
        if final_vals:
            st.write(f"Final ODE residual (mean): {final_vals[-1]:.6e}")



    # Show MSE for final frame if true solution is available
    final_frame = frames[-1]
    if final_frame.get('y_true') is not None:
        # Ensure both arrays are 1-D and comparable to avoid broadcasting issues
        y_true = np.asarray(final_frame['y_true']).flatten()
        y_pred = np.asarray(final_frame['y_pred']).flatten()
        mse = float(np.mean((y_true - y_pred) ** 2))
        st.write(f"MSE for PINN solution (final frame): {mse:.8f}")
