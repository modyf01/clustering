import numpy as np
import pytest
from kategoryzacja.clusterers import SbertUmapHdbscanClusterer


def _group_majority_accuracy(groups, predicted_labels):
    predicted_labels = np.asarray(predicted_labels)
    total = 0
    correct = 0
    for grp in groups:
        grp = np.asarray(grp)
        total += len(grp)
        if len(grp) == 0:
            continue
        vals, counts = np.unique(predicted_labels[grp], return_counts=True)
        if len(counts) == 0:
            continue
        correct += int(max(counts))
    return correct / max(1, total)


def test_model_cpu_perfect_groups(event_texts_list, expected_groups, quality_log):
    name = "sbert_umap12_hdbscan"
    model = SbertUmapHdbscanClusterer(preferred_device="cpu")
    texts = event_texts_list
    quality_log(f"Device: {model.get_device()}")
    quality_log(f"UMAP backend: {model.get_umap_backend()}")
    quality_log(f"HDBSCAN backend: {model.get_hdbscan_backend()}")
    assert "cpu" in model.get_device()
    labels = model.fit_predict(texts, n_clusters=None, random_state=42)
    acc = _group_majority_accuracy(expected_groups, labels)
    quality_log(f"[groups-acc] model={name} k={len(set(labels))}")

    quality_log(f"Model {name} (k={len(set(labels))})")
    idxs = {c: [] for c in sorted(set(labels))}
    for i, lbl in enumerate(labels):
        idxs[lbl].append(i)
    for c in sorted(idxs.keys()):
        quality_log(f"Cluster {c+1}:")
        for i in idxs[c]:
            quality_log(texts[i])
    assert len(set(labels)) == len(expected_groups)
    assert acc == 1.0


def test_model_gpu_perfect_groups(event_texts_list, expected_groups, quality_log):
    name = "sbert_umap12_hdbscan"
    model = SbertUmapHdbscanClusterer(preferred_device="cuda")
    texts = event_texts_list
    quality_log(f"Device: {model.get_device()}")
    quality_log(f"UMAP backend: {model.get_umap_backend()}")
    quality_log(f"HDBSCAN backend: {model.get_hdbscan_backend()}")
    assert "cuda" in model.get_device()
    labels = model.fit_predict(texts, n_clusters=None, random_state=42)
    acc = _group_majority_accuracy(expected_groups, labels)
    quality_log(f"model={name} k={len(set(labels))}")

    quality_log(f"Model {name} (k={len(set(labels))})")
    idxs = {c: [] for c in sorted(set(labels))}
    for i, lbl in enumerate(labels):
        idxs[lbl].append(i)
    for c in sorted(idxs.keys()):
        quality_log(f"Cluster {c+1}:")
        for i in idxs[c]:
            quality_log(texts[i])
    assert len(set(labels)) == len(expected_groups)
    assert acc == 1.0

