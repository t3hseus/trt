import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pytorch_lightning import seed_everything
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.append("../")
from os.path import join as pjoin

import torch

from src.data_generation import SPDEventGenerator, TrackParams, Vertex
from src.models.model_with_segmentation import TRTWithSegmentation
from src.normalization import ConstraintsNormalizer, TrackParamsNormalizer
from src.visualization import display_side_by_side, draw_event

MAX_EVENT_TRACKS = 5
TRUNCATION_LENGTH = 1024
BATCH_SIZE = 1
NUM_EVENTS_TRAIN = 1024
NUM_EVENTS_VALID = 1024
NUM_IMAGES = 10
PATH = r"weights\best\trt_hybrid_val.pt"

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

    model = TRTWithSegmentation(
        num_candidates=50, num_out_params=7, dropout=0.0, n_points=1024
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    vertex_dists = []
    accuracies = []
    plot_events = np.random.randint(0, num_events, num_images)
    for i in tqdm(range(num_events)):
        event, hits, labels, hits_norm, fakes_norm = generate_event(event_gen)
        inputs, hit_labels, mask = convert_event_to_batch(hits_norm, fakes_norm)

        preds = model(inputs=inputs, mask=mask)
        track_mask = torch.softmax(preds["logits"], dim=-1)[:, :, 0] > 0.5
        pred_vertex, pred_tracks = convert_preds_to_param_vertex(preds)
        pred_hits, pred_labels = generate_event_from_params(
            event_gen, pred_tracks, pred_vertex
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

        # track_distances = nearest_tracks_dist(target_tracks, pred_tracks_list)

        accuracies.append(
            (hit_labels == preds["hit_logits"].argmax(dim=-1).squeeze()).sum()
            / len(hit_labels)
        )

        if i in plot_events:
            plot(
                i,
                event_gen,
                event,
                pred_hits,
                pred_labels,
                pred_tracks,
                pred_vertex,
                track_mask,
                out_dir,
            )
    plot_histograms(accuracies, vertex_dists, out_dir)


def generate_event(event_gen):
    event = event_gen.generate_spd_event()
    hits, labels = generate_event_from_params(
        event_gen, event.track_params, event.vertex
    )
    hits_norm = ConstraintsNormalizer()(hits)
    fakes_norm = ConstraintsNormalizer()(event.fakes)
    return event, hits, labels, hits_norm, fakes_norm


def convert_event_to_batch(hits_norm, fakes_norm):
    maxlen = len(hits_norm) + len(fakes_norm)
    n_features = hits_norm.shape[-1]
    mask = np.ones(len(hits_norm))
    batch_inputs = np.zeros((1, maxlen, n_features), dtype=np.float32)
    batch_hit_labels = np.zeros((1, maxlen), dtype=bool)
    batch_mask = np.ones((1, maxlen), dtype=bool)
    # params have the fixed size - MAX_TRACKS x N_PARAMS
    batch_inputs[0, : len(hits_norm)] = hits_norm
    batch_inputs[0, len(hits_norm) :] = fakes_norm  # add fakes!
    batch_hit_labels[0, : len(hits_norm)] = mask  # hit label to check segmentation
    shuffle_idx = np.random.permutation(maxlen)
    batch_inputs[0, :] = batch_inputs[0, shuffle_idx]
    batch_hit_labels[0, :] = batch_hit_labels[0, shuffle_idx]

    inputs = torch.from_numpy(batch_inputs)
    mask = torch.from_numpy(batch_mask)
    hit_labels = torch.tensor(batch_hit_labels, dtype=torch.long).squeeze()
    return inputs, hit_labels, mask


def generate_event_from_params(event_gen, track_params, vertex):
    magnetic_field = event_gen.magnetic_field
    z_coord_range = event_gen.z_coord_range
    radii = np.linspace(
        event_gen.r_coord_range[0], event_gen.r_coord_range[1], event_gen.n_stations
    )  # mm

    hits = []
    labels = []

    for track in track_params:
        for r in radii:
            hit, _ = SPDEventGenerator.generate_hit_by_params(
                track_params=track_params[track],
                vertex=vertex,
                Rc=r,
                # magnetic_field=magnetic_field
            )

            if (hit.x, hit.y, hit.z) == (0, 0, 0):
                continue

            if not z_coord_range[0] <= hit.z <= z_coord_range[1]:
                continue

            hits.append(hit.numpy)
            labels.append(track)

    hits = np.vstack(hits, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return hits, labels


def convert_preds_to_param_vertex(preds):
    vertex_torch = preds["vertex"].squeeze()
    params = preds["params"].squeeze()
    params[:, -1] = (params[:, -1] > 0.5).to(torch.float32)
    tracks = {}
    for i, para_only in enumerate(params):
        para_t = torch.cat([vertex_torch, para_only], dim=0)
        para = (
            TrackParamsNormalizer()
            .denormalize(para_t.detach().cpu().numpy(), is_charge_categorical=True)
            .astype(float)
        )
        if i == 0:
            vertex = para[:3]
        para_obj = TrackParams(pt=para[3], phi=para[4], theta=para[5], charge=para[6])
        tracks[i] = para_obj
    vertex_obj = Vertex(x=vertex[0], y=vertex[1], z=vertex[2])
    return vertex_obj, tracks


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


def plot_histograms(accuracies, vertex_dists, out_dir) -> None:
    sns.displot(np.array(accuracies), bins=82, kde=True)
    plt.ylabel("Probability")
    plt.xlabel("Hit segmentation accuracy")
    plt.title("Hit segmentation accuracy histogram")
    plt.savefig(pjoin(out_dir, "accuracy_hist"))

    sns.displot(np.array(vertex_dists), bins=82, kde=True)
    plt.ylabel("Probability")
    plt.xlabel("Vertex distance")
    plt.title("Distances between predicted and real vertices")
    plt.savefig(pjoin(out_dir, "vertex_hist"))
    print(
        f"Hit accuracy: {np.mean(accuracies)}, mean vertex distance {np.mean(vertex_dists)}"
    )


def match_targets(outputs, targets):
    cost_matrix = torch.cdist(outputs, targets, p=1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return row_ind, col_ind


def cardinality_error(pred_tracks, target_tracks):
    return len(pred_tracks) - len(target_tracks)


def nearest_tracks_dist(gt_tracks, pred_tracks):
    """
    Find the nearest predicted track for each ground truth track, and compute the distance between them.

    Args:
        gt_tracks (list of torch.Tensor): Ground truth tracks, each of shape (length_of_track_gt, 3).
        pred_tracks (list of torch.Tensor): Predicted tracks, each of shape (length_of_track_pred, 3).

    Returns:
        distances (torch.Tensor): Pairwise distances between nearest ground truth and predicted tracks.
                                  Shape: (num_gt_tracks,)
        nearest_indices (torch.Tensor): Indices of the nearest predicted tracks for each ground truth track.
                                        Shape: (num_gt_tracks,)
    """
    num_gt_tracks = len(gt_tracks)
    num_pred_tracks = len(pred_tracks)

    # Initialize the tensor to store distances between tracks
    distances = torch.zeros(num_gt_tracks, dtype=torch.float)

    max_distances = torch.zeros(num_gt_tracks, dtype=torch.float)
    nearest_indices = torch.zeros(num_gt_tracks, dtype=torch.long)

    # Iterate through each ground truth track
    for i, gt_track in enumerate(gt_tracks):
        length_gt = gt_track.shape[0]  # Number of points in current ground truth track

        # Track to store distances to each predicted track
        total_distances = torch.empty(num_pred_tracks)
        max_t_distances = torch.empty(num_pred_tracks)
        # Compare the current ground truth track with all predicted tracks
        for j, pred_track in enumerate(pred_tracks):
            length_pred = pred_track.shape[0]  # Number of points in the predicted track

            # Determine the minimum length between the two tracks to compare their points
            min_length = min(length_gt, length_pred)
            # Select the first 'min_length' points from both tracks
            for k, z in enumerate(pred_track[:, 2]):
                if k < len(gt_track):
                    hit_dist = (
                        np.abs(gt_track[k, 0] - pred_track[k, 1])
                        + np.abs(gt_track[k, 1] - pred_track[k, 1])
                        + np.abs(gt_track[k, 2] - pred_track[k, 2])
                    )
                    total_distances[j] += hit_dist
                    max_t_distances[j] = max(max_t_distances[j], hit_dist)

        # Find the index of the nearest predicted track
        nearest_idx = torch.argmin(total_distances)
        nearest_indices[i] = nearest_idx
        distances[i] = total_distances[nearest_idx]
        max_distances[i] = max_t_distances[nearest_idx]
    return {
        "dist": distances,
        "max_hit_dist": max_distances,
        "near_index": nearest_indices,
    }


if __name__ == "__main__":
    inference()
