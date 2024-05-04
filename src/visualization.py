import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from typing import Tuple
from .data_generation import Event


def draw_event(
    event: Event,
    x_coord_range: Tuple[float, float] = (-851., 851.),
    y_coord_range: Tuple[float, float] = (-851., 851.),
    z_coord_range: Tuple[float, float] = (-2386., 2386.),
    colorscale: str = "Plotly3"
) -> go.Figure:

    fig = go.Figure()
    uniq_tracks = np.unique(event.track_ids)
    colors = sample_colorscale(colorscale, uniq_tracks / uniq_tracks.max())

    for i, label in enumerate(uniq_tracks):
        track_hits = event.hits[event.track_ids == label]
        fig.add_trace(go.Scatter3d(
            x=track_hits[:, 0],
            y=track_hits[:, 1],
            z=track_hits[:, 2],
            marker=dict(
                size=1,
                color=colors[i],
            ),
            mode="markers",
            name=f"Event #{label}"
        ))

    # draw vertex
    fig.add_trace(go.Scatter3d(
        x=[event.vertex.x],
        y=[event.vertex.y],
        z=[event.vertex.z],
        marker=dict(
            size=2,
            color="red",
        ),
        mode="markers",
        name="Vertex"
    ))

    # draw fakes
    fig.add_trace(go.Scatter3d(
        x=event.fakes[:, 0],
        y=event.fakes[:, 1],
        z=event.fakes[:, 2],
        marker=dict(
            size=1,
            color="gray",
        ),
        opacity=0.35,
        mode="markers",
        name="Fakes",
    ))

    fig.update_layout(
        margin=dict(
            t=20,
            b=10,
            l=10,
            r=10
        ),
        scene=dict(
            xaxis=dict(range=x_coord_range),
            yaxis=dict(range=y_coord_range),
            zaxis=dict(range=z_coord_range),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        scene_camera=dict(
            eye=dict(x=2, y=0.1, z=0.1)
        )
    )

    return fig
