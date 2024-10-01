from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots

from .constants import OX_RANGE, OY_RANGE, OZ_RANGE
from .data_generation import Array3, ArrayN, ArrayNx3


def draw_event(
    hits: ArrayNx3[np.float32],
    labels: ArrayN[np.int32],
    vertex: Optional[Array3[np.float32]] = None,
    fakes: Optional[ArrayNx3[np.float32]] = None,
    x_coord_range: Tuple[float, float] = OX_RANGE,
    y_coord_range: Tuple[float, float] = OY_RANGE,
    z_coord_range: Tuple[float, float] = OZ_RANGE,
    colorscale: str = "Plotly3",
) -> go.Figure:

    fig = go.Figure()
    uniq_tracks = np.unique(labels)
    colors = sample_colorscale(colorscale, uniq_tracks / uniq_tracks.max())

    for i, label in enumerate(uniq_tracks):
        track_hits = hits[labels == label]
        fig.add_trace(
            go.Scatter3d(
                x=track_hits[:, 0],
                y=track_hits[:, 1],
                z=track_hits[:, 2],
                marker=dict(
                    size=1,
                    color=colors[i],
                ),
                mode="markers",
                name=f"Track #{label}",
            )
        )

    # draw vertex
    if vertex is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[vertex[0]],
                y=[vertex[1]],
                z=[vertex[2]],
                marker=dict(
                    size=2,
                    color="red",
                ),
                mode="markers",
                name="Vertex",
            )
        )

    # draw fakes
    if fakes is not None:
        fig.add_trace(
            go.Scatter3d(
                x=fakes[:, 0],
                y=fakes[:, 1],
                z=fakes[:, 2],
                marker=dict(
                    size=1,
                    color="gray",
                ),
                opacity=0.35,
                mode="markers",
                name="Fakes",
            )
        )

    fig.update_layout(
        margin=dict(t=20, b=10, l=10, r=10),
        scene=dict(
            xaxis=dict(range=x_coord_range),
            yaxis=dict(range=y_coord_range),
            zaxis=dict(range=z_coord_range),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        scene_camera=dict(eye=dict(x=2, y=0.1, z=0.1)),
    )

    return fig


def display_side_by_side(
    predicted_event_fig: go.Figure,
    original_event_fig: go.Figure,
    left_title: str = "Prediction",
    right_title: str = "Ground Truth"
) -> go.Figure:

    # Create a 1x2 subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(left_title, right_title)
    )

    # Get traces from the left figure and add them to the first subplot
    for trace in predicted_event_fig.data:
        trace_left = trace.to_plotly_json()
        trace_left['name'] = f"[pred] {trace_left['name']}"
        trace_left['showlegend'] = True
        fig.add_trace(go.Scatter3d(**trace_left), row=1, col=1)

    # Get traces from the right figure and add them to the second subplot
    for trace in original_event_fig.data:
        trace_right = trace.to_plotly_json()
        trace_right['name'] = f"[orig] {trace_right['name']}"
        trace_right['showlegend'] = True
        fig.add_trace(go.Scatter3d(**trace_right), row=1, col=2)

    # Update layout for the overall figure
    fig.update_layout(
        margin=dict(t=50, b=10, l=10, r=10),
        height=600
    )

    # Apply the camera and aspect ratio settings to both scenes
    fig.update_scenes(
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(eye=dict(x=2, y=0.1, z=0.1)),
        row=1, col=1  # Apply to the first subplot (Prediction)
    )

    fig.update_scenes(
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(eye=dict(x=2, y=0.1, z=0.1)),
        row=1, col=2  # Apply to the second subplot (Ground Truth)
    )

    return fig
