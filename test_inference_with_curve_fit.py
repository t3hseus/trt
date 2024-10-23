import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pytorch_lightning import seed_everything
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from test_inference import match_targets, generate_event_from_params, convert_preds_to_param_vertex, \
    convert_event_to_batch, generate_event, get_params_dists, get_tracks_dists

sys.path.append("../")
from os.path import join as pjoin

import torch
from torch.nn import functional as F
from src.data_generation import SPDEventGenerator
from src.model import TRTHybrid
from src.visualization import display_side_by_side, draw_event


MAX_EVENT_TRACKS = 5
TRUNCATION_LENGTH = 1024
BATCH_SIZE = 1
NUM_EVENTS_VALID = 1024
NUM_IMAGES = 10
PATH = r"weights\server_night\trt_hybrid_val.pt"

seed_everything(13)


def inference(
    weights_path: str = PATH,
    num_events: int = NUM_EVENTS_VALID,
    truncation_length: int = TRUNCATION_LENGTH,
    max_event_tracks: int = MAX_EVENT_TRACKS,
    num_images: int = NUM_IMAGES,
    result_dir: str = "plots",
) -> None:
    out_dir = pjoin(
        result_dir, datetime.today().strftime("%Y-%m-%d"), PATH.split("\\")[-2]
    )
    os.makedirs(out_dir, exist_ok=True)

    event_gen = SPDEventGenerator(
        generate_fixed_tracks_num=False,
        detector_eff=0.98,
        max_event_tracks=max_event_tracks,
    )

    model = TRTHybrid(
        num_candidates=5, num_out_params=7, dropout=0.0, num_points=truncation_length, zero_based_decoder=False
    )
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    vertex_dists = []
    accuracies = []
    params_distances = {i: [] for i in ["pt", "phi", "theta", "charge"]}

    fitted_distances = []

    tracks_distances = []
    plot_events = np.random.randint(0, num_events, num_images)
    for ev in tqdm(range(num_events)):
        event, hits, labels, hits_norm, fakes_norm, track_params = generate_event(event_gen)
        inputs, hit_labels, mask = convert_event_to_batch(hits_norm, fakes_norm)

        preds = model(inputs, mask=mask)
        track_mask = torch.softmax(preds["logits"], dim=-1)[:, :, 0] > 0.8
        print("Selected tracks: ", (~track_mask).sum())
        pred_vertex, pred_tracks = convert_preds_to_param_vertex(preds)
        pred_hits, pred_labels = generate_event_from_params(
            event_gen, pred_tracks, event.vertex
        )
        vertex_dists.append(
            np.abs(pred_vertex.x - event.vertex.x)
            + np.abs(pred_vertex.y - event.vertex.y)
            + np.abs(pred_vertex.z - event.vertex.z)
        )
        target_tracks = []
        for label in np.unique(labels):
            if label == -1:
                continue
            target_tracks.append(torch.tensor(hits[labels == label]))
        pred_tracks_list = []
        for label in np.unique(pred_labels):
            pred_tracks_list.append(torch.tensor(pred_hits[pred_labels == label]))

        param_distance = get_params_dists(pred_tracks, track_params)
        for i, v in param_distance.items():
            params_distances[i].append(v)
        fitted_param_dists = get_fitted_params_dists(pred_tracks_list, target_tracks)
        fitted_distances.extend(fitted_param_dists)
        track_distance = get_tracks_dists(pred_tracks, track_params, pred_tracks_list, target_tracks)
        tracks_distances.extend(track_distance)

        accuracies.append(
            (hit_labels == preds["hit_logits"].argmax(dim=-1).squeeze()).sum()
            / len(hit_labels)
        )

        if ev in plot_events:
            plot(
                ev,
                event_gen,
                event,
                pred_hits,
                pred_labels,
                pred_tracks,
                pred_vertex,
                track_mask,
                out_dir,
            )
    plot_histograms(
        accuracies, vertex_dists, params_distances, tracks_distances,  fitted_distances, out_dir
    )


def get_fitted_params_dists(pred_tracks, target_tracks):
    p_pred = []
    p_target = []
    for track in pred_tracks:
        curve = get_curve(track.cpu().numpy())
        if curve is not None:
            p_pred.append(curve)
    for track in target_tracks:
        curve = get_curve(track.cpu().numpy())
        if curve is not None:
            p_target.append(curve)
    pred_vectors = torch.tensor(p_pred)
    target_vectors = torch.tensor(p_target)

    if len(pred_vectors.shape) == 1:
        pred_vectors = pred_vectors.unsqueeze(0)
    if len(target_vectors.shape) == 1:
        target_vectors = target_vectors.unsqueeze(0)
    if target_vectors.shape[1] == 0 or pred_vectors.shape[1] == 0:
        return [1000.]
    row_ind, col_ind = match_targets(
        outputs=pred_vectors,
        targets=target_vectors,
    )
    matched_outputs = pred_vectors[row_ind]
    matched_targets = target_vectors[col_ind]
    outputs = []
    for track_num in range(len(matched_targets)):
        outputs.append(F.l1_loss(
            matched_outputs[track_num],
            matched_targets[track_num]
        ).item())
    #print(f"Matched by params | preds: {row_ind} and targets: {col_ind}")
    return outputs


def plot(
    i,
    event_gen,
    event,
    pred_hits,
    pred_labels,
    pred_tracks,
    pred_vertex,
    track_mask,
    out_dir,
):
    real_event = draw_event(
        hits=event.hits,
        fakes=event.fakes,
        vertex=event.vertex.numpy,
        labels=event.track_ids,
    )
    pred_event = draw_event(
        hits=pred_hits,
        fakes=None,
        vertex=event.vertex.numpy,
        labels=pred_labels,
    )
    side_by_side = display_side_by_side(
        predicted_event_fig=pred_event,
        original_event_fig=real_event,
        left_title="Prediction",
        right_title="Ground Truth",
    )
    side_by_side.write_html(pjoin(out_dir, str(i) + "_side_by_side.html"))

    pred_event = draw_event(
        hits=pred_hits,
        fakes=None,
        vertex=pred_vertex.numpy,
        labels=pred_labels,
    )
    side_by_side = display_side_by_side(
        predicted_event_fig=pred_event,
        original_event_fig=real_event,
        left_title="Prediction",
        right_title="Ground Truth",
    )
    side_by_side.write_html(pjoin(out_dir, str(i) + "_side_by_side_pred_vertex.html"))

    filtered_pred_tracks = {}
    track_mask = track_mask.squeeze()
    for i, track in pred_tracks.items():
        if track_mask[i]:
            filtered_pred_tracks[i] = track
    if filtered_pred_tracks:
        filtered_pred_hits, filtered_pred_labels = generate_event_from_params(
            event_gen, filtered_pred_tracks, pred_vertex
        )
        filtered_pred_event = draw_event(
            hits=filtered_pred_hits,
            fakes=None,
            vertex=event.vertex.numpy,
            labels=filtered_pred_labels,
        )
        side_by_side = display_side_by_side(
            predicted_event_fig=filtered_pred_event,
            original_event_fig=real_event,
            left_title="Filtered Prediction",
            right_title="Ground Truth",
        )
        side_by_side.write_html(pjoin(out_dir, str(i) + "_side_by_syde_filtered.html"))


def plot_histograms(
        accuracies, vertex_dists, param_distances, track_distances, fitted_distances, out_dir
) -> None:
    sns.displot(np.array(accuracies), bins=82, kde=True)
    plt.ylabel("Probability")
    plt.xlabel("Hit segmentation accuracy")
    plt.title("Hit segmentation accuracy histogram")
    plt.savefig(pjoin(out_dir, "accuracy_hist"),  bbox_inches='tight')

    sns.displot(np.array(vertex_dists), bins=82, kde=True)
    plt.ylabel("Probability")
    plt.xlabel("Vertex distance")
    plt.title("Distances between predicted and real vertices")
    plt.savefig(pjoin(out_dir, "vertex_hist"),  bbox_inches='tight')
    for i, dist in param_distances.items():
        sns.displot(np.array(dist), bins=82, kde=True)
        plt.ylabel("Probability")
        plt.xlabel("Per-param distance")
        plt.title(f"Distance between predicted and real {i} (l1)")
        plt.savefig(pjoin(out_dir, f"params_hist_{i}"),  bbox_inches='tight')
        print(f"Mean {i} param distance: {np.mean(dist)}")
        print(f"Median {i} param distance: {np.median(dist)}")
    sns.displot(np.array(track_distances), bins=82, kde=True)
    plt.ylabel("Probability")
    plt.xlabel("Per-param distance")
    plt.title("Distances between predicted and real tracks (l1)")
    plt.savefig(pjoin(out_dir, f"track_distances"), bbox_inches='tight')
    sns.displot(np.array(fitted_distances), bins=82, kde=True)
    plt.ylabel("Probability")
    plt.xlabel("Distance between fitted curve params")
    plt.title("Distances between fitted curve params for predicted and real tracks (l1)")
    plt.savefig(pjoin(out_dir, f"fitted_curve_params_dist"), bbox_inches='tight')
    print(f"Mean track distance: {np.mean(track_distances)}")
    print(f"Mean fitted params distance: {np.mean(fitted_distances)}")
    print(
        f"Hit accuracy: {np.mean(accuracies)}, mean vertex distance {np.mean(vertex_dists)}"
    )
    print(
        f"Hit median accuracy: {np.median(accuracies)}, median vertex distance {np.median(vertex_dists)}"
    )


def get_params_diffs(
        pred_vectors,
        target_vectors,
        param_names: list[str] = ["pt", "phi", "theta", "charge"],
):
    row_ind, col_ind = match_targets(
                outputs=pred_vectors,
                targets=target_vectors,
            )
    matched_outputs = pred_vectors[row_ind]
    matched_targets = target_vectors[col_ind]
    outputs = {}
    for param_num in range(matched_targets.shape[-1]):
        outputs[param_names[param_num]] = F.l1_loss(
                matched_outputs[:, param_num],
                matched_targets[:, param_num]
            ).item()
    return outputs, row_ind, col_ind


def get_curve(hits):
    """data is a 3d array of shape [n_hits, 3]
    outputs 9 fitted coefficients
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from numpy.polynomial import Polynomial
    t = np.linspace(-1, 1, len(hits))

    # curve fit function
    def func(t, x2, x1, x0, y2, y1, y0, z2, z1, z0):
        Px = Polynomial([x2, x1, x0])
        Py = Polynomial([y2, y1, y0])
        Pz = Polynomial([z2, z1, z0])
        return np.concatenate([Px(t), Py(t), Pz(t)])

    start_vals = [1, 1, 1,
                  1, 1, 1,
                  -1, -1, -1]
    xyz = np.concatenate([hits[:, 0], hits[:, 1], hits[:, 2]])
    # xyz = data.flatten()
    if len(xyz) > 9:
        popt, _ = curve_fit(func, t, xyz, p0=start_vals)
    else:
        return None
    return popt


if __name__ == "__main__":
    inference()
