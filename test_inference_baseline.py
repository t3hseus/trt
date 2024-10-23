import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pytorch_lightning import seed_everything
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.model import TRTBaseline

sys.path.append("../")
from os.path import join as pjoin

import torch
from torch.nn import functional as F
from src.data_generation import SPDEventGenerator, TrackParams, Vertex
from src.normalization import HitsNormalizer, TrackParamsNormalizer
from src.visualization import display_side_by_side, draw_event

MAX_EVENT_TRACKS = 5
TRUNCATION_LENGTH = 1024
BATCH_SIZE = 1
NUM_EVENTS_VALID = 1024
NUM_IMAGES = 10
PATH = r"weights\baseline\trt_hybrid_val.pt"

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

    model = TRTBaseline()
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    vertex_dists = []
    accuracies = []
    params_distances = {i: [] for i in ["pt", "phi", "theta", "charge"]}
    tracks_distances = []
    plot_events = np.random.randint(0, num_events, num_images)
    true_clusters_num = 0
    pred_clusters_num = 0
    all_tracks_num = 0
    clusters_accuracies_vs_number = {i: [] for i in range(6)}
    fitted_distances = []
    for ev in tqdm(range(num_events)):
        event, hits, fakes, labels, hits_norm, fakes_norm, track_params = generate_event(event_gen)
        inputs, hit_labels, mask, hits, track_labels = convert_event_to_batch(hits_norm, fakes_norm, hits, fakes, labels)

        preds = model(inputs, mask=mask)

        target_tracks = []
        num_tracks = 0
        for label in np.unique(track_labels):
            if label == -1:
                continue
            target_tracks.append(torch.tensor(hits[track_labels == label]))
            num_tracks += 1
        accuracies.append(
            (hit_labels == (preds["hit_logits"].sigmoid() > 0.5).squeeze(-1)).sum()
            / len(hit_labels)
        )
        selection_mask = (preds["hit_logits"].sigmoid() > 0.5).squeeze()
        selected_hits = hits[selection_mask]
        selected_track_labels = track_labels[selection_mask]
        correct_clusters = 0
        all_clusters = 0
        clustered_hits = []
        clusters = []
        pred_clusters = preds["clusters"].squeeze()
        for cluster in torch.unique(pred_clusters):
            if cluster == -1:
                continue
            clustered_hits.append(selected_hits[pred_clusters==cluster])
            clusters.append(pred_clusters[pred_clusters==cluster].cpu().numpy())
            target_track_labels = selected_track_labels[pred_clusters==cluster]
            target_track_ids, target_track_nums = torch.unique(torch.tensor(target_track_labels), return_counts=True)
            target_id = target_track_ids[0].item()
            found_target_part = target_track_nums[0].item()
            all_clusters += 1
            try:
                if found_target_part > len(track_labels[target_id]) * 0.8:
                    correct_clusters += 1
            except TypeError:
                # If only one
                continue
            clusters_accuracies_vs_number[num_tracks].append(correct_clusters/all_clusters)

        true_clusters_num += correct_clusters
        pred_clusters_num += all_clusters
        all_tracks_num += num_tracks

        fitted_param_dists = get_fitted_params_dists(clustered_hits,target_tracks)
        fitted_distances.extend(fitted_param_dists)
        if ev in plot_events:
            plot(
                ev,
                event,
                np.concatenate(clustered_hits),
                np.concatenate(clusters),
                out_dir,
            )
    print("Precision in track prediction: ", true_clusters_num / pred_clusters_num)
    print("Recall in track prediction: ", true_clusters_num / all_tracks_num)
    plot_histograms(accuracies, fitted_distances, out_dir)


def get_params_dists(pred_tracks, target_tracks):
    pred_unnorm_params = convert_params_to_tensor(pred_tracks)
    target_unnorm_params = convert_params_to_tensor(target_tracks)
    diffs, row_ind, col_ind = get_params_diffs(
        pred_unnorm_params, target_unnorm_params
    )
    #print(f"Matched by params | preds: {row_ind} and targets: {col_ind}")
    return diffs

def get_fitted_params_dists(pred_tracks, target_tracks):
    p_pred = []
    p_target = []
    for track in pred_tracks:
        curve = get_curve(track)
        if curve is not None:
            p_pred.append(curve)
    for track in target_tracks:
        curve = get_curve(track)
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


def get_tracks_dists(pred_params, target_params, pred_tracks, target_tracks):
    #pred_vectors = torch.stack(pred_tracks)
    #target_vectors = torch.stack(target_tracks)
    pred_unnorm_params = convert_params_to_tensor(pred_params)
    target_unnorm_params = convert_params_to_tensor(target_params)
    row_ind, col_ind = match_targets(
        outputs=pred_unnorm_params,
        targets=target_unnorm_params,
    )
    try:
        matched_outputs = [pred_tracks[i] for i in row_ind]
        matched_targets = [target_tracks[j] for j in col_ind]
    except:
        return []
    outputs = []
    for track_num in range(len(matched_targets)):
        outputs.append(F.l1_loss(
            matched_outputs[track_num],
            matched_outputs[track_num]
        ).item())
    #print(f"Matched by params | preds: {row_ind} and targets: {col_ind}")
    return outputs


def convert_params_to_tensor(params: dict[int, TrackParams]):
    torch_params = torch.zeros((len(params), 4), dtype=torch.float32)
    for i, track in params.items():
        torch_params[i, :] = track.torch
    return torch_params


def generate_event(event_gen):
    event = event_gen.generate_spd_event()
    hits, labels = generate_event_from_params(
        event_gen, event.track_params, event.vertex
    )
    hits_norm = HitsNormalizer()(hits)
    fakes_norm = HitsNormalizer()(event.fakes)

    return event, hits, event.fakes, labels, hits_norm, fakes_norm, event.track_params


def convert_event_to_batch(hits_norm, fakes_norm, hits, fakes, labels):
    maxlen = len(hits_norm) + len(fakes_norm)
    hits = np.concatenate((hits, fakes), axis=0)
    n_features = hits_norm.shape[-1]
    mask = np.ones(len(hits_norm))
    batch_inputs = np.zeros((1, maxlen, n_features), dtype=np.float32)
    batch_hit_labels = np.zeros((1, maxlen), dtype=bool)
    unnorm_labels = np.ones(maxlen, dtype=bool)*(-1)
    batch_mask = np.ones((1, maxlen), dtype=bool)
    batch_params = np.zeros((1, maxlen, n_features), dtype=np.float32)
    # params have the fixed size - MAX_TRACKS x N_PARAMS
    batch_inputs[0, : len(hits_norm)] = hits_norm
    batch_inputs[0, len(hits_norm) :] = fakes_norm  # add fakes!
    batch_hit_labels[0, : len(hits_norm)] = mask  # hit label to check segmentation
    unnorm_labels[:len(hits_norm)] = labels
    shuffle_idx = np.random.permutation(maxlen)
    batch_inputs[0, :] = batch_inputs[0, shuffle_idx]
    batch_hit_labels[0, :] = batch_hit_labels[0, shuffle_idx]
    unnorm_labels = unnorm_labels[shuffle_idx]
    hits = hits[shuffle_idx]

    inputs = torch.from_numpy(batch_inputs)
    mask = torch.from_numpy(batch_mask)
    hit_labels = torch.tensor(batch_hit_labels, dtype=torch.long).squeeze()
    return inputs, hit_labels, mask, hits, unnorm_labels


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
    event,
    pred_hits,
    pred_labels,
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


def plot_histograms(accuracies, fitted_dists, out_dir) -> None:
    sns.displot(np.array(accuracies), bins=82, kde=True)
    plt.ylabel("Probability")
    plt.xlabel("Hit segmentation accuracy")
    plt.title("Hit segmentation accuracy histogram")
    plt.savefig(pjoin(out_dir, "accuracy_hist"),  bbox_inches='tight')
    sns.displot(np.array(fitted_dists), bins=82, kde=True)
    plt.ylabel("Probability")
    plt.xlabel("Histogram of fitted_dists")
    plt.title("Histogram of distances between predicted and gt fitted curve params")
    plt.savefig(pjoin(out_dir, "fitted_distances_hist"), bbox_inches='tight')
    print(f"Mean fitted distance: {np.mean(fitted_dists)}")


def match_targets(outputs, targets):
    cost_matrix = torch.cdist(outputs, targets, p=1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return row_ind, col_ind


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
