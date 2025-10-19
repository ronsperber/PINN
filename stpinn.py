import streamlit as st
import importlib
import torch
import numpy as np
import plotly.graph_objects as go
import time

from pinn_utils import pinn

importlib.reload(pinn)


st.title("PINN ODE Solver")

# Full ODE metadata mapping for the sidebar and solver logic
from pinn_utils.ode_meta import ODES

# Sidebar inputs driven by ODES metadata
ode_choice = st.sidebar.selectbox("Choose ODE", list(ODES.keys()))
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
    if torch.is_tensor(val):
        return val.detach().cpu().numpy().tolist()  # convert to list
    elif isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    else:
        return val

# Detect sidebar parameter changes and clear previous frames if any parameter changed
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
    activation=activation
    )
if 'last_params' not in st.session_state:
    st.session_state['last_params'] = current_params
elif st.session_state['last_params'] != current_params:
    # user changed parameters — clear any previously computed frames so chart doesn't persist
    st.session_state.pop('frames', None)
    st.session_state.pop('x_np', None)
    st.session_state.pop('png_bytes', None)
    st.session_state.pop('gif_bytes', None)
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
        # solve the DE
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
                ck_fn = checkpoint[1]
                nn_for_eval = _nn_from_checkpoint_fn(ck_fn)
                if nn_for_eval is None:
                    # final frame: use the trained NN instance
                    nn_for_eval = NN

                # build a differentiable trial function from this NN so we can compute derivatives/residual
                y_fn = pinn.get_y_trial(x0, ics, nn_for_eval)
                # ensure x_train requires grad for derivative computation
                x_for_eval = x_train.detach().clone().requires_grad_(True)
                # compute the PDE loss at each checkpoint
                y_torch = y_fn(x_for_eval)
                derivs = pinn.derivatives(y_torch, x_for_eval, len(ics))
                try:
                    res = F(x_for_eval, *derivs)
                    pde_loss = float(torch.mean(res**2).item())
                except Exception as e:
                    pde_loss = None

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
                frames.append({"y_pred": y_pred, "y_true": y_true, "title": title, "epoch": epoch_val, "pde_loss": pde_loss})

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
            data = [go.Scatter(x=x, y=fr['y_pred'].flatten())]
        annotations = []
        if fr['y_true'] is not None:
            if meta.get("is_system", False):
                data.append(go.Scatter(x=fr['y_true'][:,0].flatten(), y=fr['y_true'][:,1].flatten(), mode='lines', name= 'True Solution', line=dict(dash='dash')))
            else:
                data.append(go.Scatter(x=x, y=fr['y_true'].flatten()))
            mse = np.mean((fr['y_pred'].flatten() - fr['y_true'].flatten())**2)
            ann_text = f"MSE: {mse:.6f}"
            # include PDE residual when available
            pde_val = fr.get('pde_loss')
            if pde_val is not None:
                ann_text += f"<br>Residual: {pde_val:.3e}"
            annotations.append(dict(
                xref='paper', yref='paper', x=0.95, y=0.95,  # top-right corner
                text=ann_text,
                showarrow=False,
                font=dict(size=12, color="black")
            ))
        else:
            # no true solution; still show residual if available
            pde_val = fr.get('pde_loss')
            if pde_val is not None:
                annotations.append(dict(
                    xref='paper', yref='paper', x=0.95, y=0.95,
                    text=f"Residual: {pde_val:.3e}",
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

    # Show PDE residuals (mean squared residual) over frames if available using Plotly
    # user-controlled toggle to show/hide residuals
    show_residuals = st.sidebar.checkbox("Show PDE residuals", value=False)
    pde_losses = [fr.get('pde_loss') for fr in frames]
    if show_residuals and any(pl is not None for pl in pde_losses):
        # convert None -> nan for plotting
        yvals = [pl if pl is not None else float('nan') for pl in pde_losses]
        res_fig = go.Figure()
        res_fig.add_trace(go.Scatter(x=list(range(1, len(yvals) + 1)), y=yvals, mode='lines+markers', name='PDE residual'))
        res_fig.update_layout(title='PDE residual (per-frame)', xaxis_title='Checkpoint', yaxis_title='Mean PDE residual', template='plotly_white')
        st.plotly_chart(res_fig, use_container_width=True)
        # show final numeric value if available
        final_vals = [v for v in pde_losses if v is not None]
        if final_vals:
            st.write(f"Final PDE residual (mean): {final_vals[-1]:.6e}")

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
        # Ensure both arrays are 1-D and comparable to avoid broadcasting issues
        y_true = np.asarray(final_frame['y_true']).flatten()
        y_pred = np.asarray(final_frame['y_pred']).flatten()
        mse = float(np.mean((y_true - y_pred) ** 2))
        st.write(f"MSE for PINN solution (final frame): {mse:.8f}")