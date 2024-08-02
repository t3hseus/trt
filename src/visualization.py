from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

from .constants import OX_RANGE, OY_RANGE, OZ_RANGE
from .data_generation import Array3, ArrayN, ArrayNx3


def draw_event(
    hits: ArrayNx3[np.float32],
    vertex: Array3[np.float32],
    labels: ArrayN[np.int32],
    fakes: Optional[ArrayNx3[np.float32]] = None,
    predicted_hits: Optional[ArrayNx3[np.float32]] = None,
    predicted_tracks: Optional[ArrayNx3[np.float32]] = None,
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
                name=f"Event #{label}",
            )
        )

    # draw vertex
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
    if predicted_tracks is not None:
        pred_colors = sample_colorscale(
            colorscale, predicted_tracks / predicted_tracks.max()
        )

        for i, label in enumerate(predicted_tracks):
            track_hits = predicted_hits[i]
            fig.add_trace(
                go.Scatter3d(
                    x=track_hits[:, 0],
                    y=track_hits[:, 1],
                    z=track_hits[:, 2],
                    marker=dict(
                        size=1,
                        color=pred_colors[i],
                    ),
                    opacity=0.7,
                    mode="markers",
                    name=f"Predicted Event #{label}",
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


def draw_event_with_preds(
    hits: ArrayNx3[np.float32],
    predicted_hits: ArrayNx3[np.float32],
    vertex: Array3[np.float32],
    labels: ArrayN[np.int32],
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
                name=f"Event #{label}",
            )
        )

    # draw vertex
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
