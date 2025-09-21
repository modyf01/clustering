from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .text_preprocessing import preprocess_many


def _compute_centers_from_labels(embeddings, labels: Sequence[int], n_clusters: int, xp):
	centers = []
	X = embeddings
	labels_xp = xp.asarray(labels)
	for c in range(n_clusters):
		mask = labels_xp == c
		if xp.any(mask):
			centers.append(X[mask].mean(axis=0))
		else:
			centers.append(X[0])
	return xp.vstack(centers)


def _pairwise_squared_euclidean(A, B, xp):
	AA = xp.sum(A * A, axis=1, keepdims=True)
	BB = xp.sum(B * B, axis=1, keepdims=True).T
	return AA + BB - 2.0 * (A @ B.T)



class BaseTextClusterer:
	is_encoder_based: bool = False

	def embed(self, texts: Sequence[str]):
		raise NotImplementedError

	def fit_predict(self, texts: Sequence[str], n_clusters: Optional[int], random_state: Optional[int] = None) -> List[int]:
		raise NotImplementedError


class _SbertBase(BaseTextClusterer):
	is_encoder_based: bool = True

	def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", preferred_device: Optional[str] = None) -> None:
		self._model_name = model_name
		self._preferred_device = preferred_device
		self._model = None
		self._xp = None
		self._cp = None
		self._UMAP = None
		self._HDBSCAN = None
		self._umap_backend = None
		self._hdbscan_backend = None

	def _select_backends(self) -> None:
		if self._UMAP is not None and self._HDBSCAN is not None and self._xp is not None:
			return
		dev = (self._preferred_device or "cpu").strip().lower()
		if dev in ("cuda", "gpu") or dev.startswith("cuda:"):
			import torch
			if not torch.cuda.is_available():
				raise RuntimeError("CUDA requested but not available")
			try:
				import cupy as cp
				from cuml.manifold.umap import UMAP as UMAP_CU
				from cuml.cluster.hdbscan import HDBSCAN as HDBSCAN_CU
			except Exception:
				raise RuntimeError("GPU pipeline requested but cuML/CuPy backends are not available")
			self._xp = cp
			self._cp = cp
			self._UMAP = UMAP_CU
			self._HDBSCAN = HDBSCAN_CU
			self._umap_backend = "cuml"
			self._hdbscan_backend = "cuml"
		else:
			from umap import UMAP as UMAP_CPU
			from sklearn.cluster import HDBSCAN as HDBSCAN_CPU
			self._xp = np
			self._cp = None
			self._UMAP = UMAP_CPU
			self._HDBSCAN = HDBSCAN_CPU
			self._umap_backend = "umap-learn"
			self._hdbscan_backend = "sklearn"

	def _ensure_model(self):
		self._select_backends()
		if self._model is None:
			device_arg = None
			if isinstance(self._preferred_device, str):
				dev = self._preferred_device.strip().lower()
				if dev in ("cuda", "gpu") or dev.startswith("cuda:"):
					device_arg = dev if dev.startswith("cuda:") else "cuda"
				elif dev == "cpu":
					device_arg = "cpu"
			self._model = SentenceTransformer(self._model_name, device=device_arg)

	def get_device(self) -> str:
		self._ensure_model()
		device = getattr(self._model, "device", None)
		return str(device) if device is not None else "cpu"

	def get_umap_backend(self) -> str:
		self._select_backends()
		return str(self._umap_backend)

	def get_hdbscan_backend(self) -> str:
		self._select_backends()
		return str(self._hdbscan_backend)

	def embed(self, texts: Sequence[str]):
		self._ensure_model()
		if not isinstance(texts, (list, tuple)):
			texts = list(texts)
		texts = preprocess_many(texts)
		X_t = self._model.encode(
			texts,
			show_progress_bar=False,
			convert_to_tensor=True,
			normalize_embeddings=True,
		)
		if X_t.is_cuda:
			if self._cp is None:
				raise RuntimeError("CUDA embeddings require CuPy for downstream GPU pipeline")
			from torch.utils.dlpack import to_dlpack
			return self._cp.from_dlpack(to_dlpack(X_t))
		return X_t.detach().cpu().numpy()


class SbertUmapHdbscanClusterer(_SbertBase):
	def _make_umap(self, n_components: int, n_neighbors: int, random_state: Optional[int]):
		if self._umap_backend == "cuml":
			return self._UMAP(
				n_components=n_components,
				n_neighbors=n_neighbors,
				min_dist=0.0,
				metric="cosine",
				init="spectral",
				output_type="cupy",
			)
		seed = (42 if random_state is None else random_state)
		return self._UMAP(
			n_components=n_components,
			n_neighbors=n_neighbors,
			min_dist=0.0,
			metric="cosine",
			random_state=seed,
			init="spectral",
		)

	def _make_hdbscan(self):
		return self._HDBSCAN(
			min_cluster_size=3,
			min_samples=2,
			metric="euclidean",
			cluster_selection_method="eom",
			cluster_selection_epsilon=0.0,
		)

	def fit_predict(self, texts: Sequence[str], n_clusters: Optional[int] = None, random_state: Optional[int] = None) -> List[int]:
		if not isinstance(texts, (list, tuple)):
			texts = list(texts)
		if len(texts) == 0:
			return []
		X_embed = self.embed(texts)
		n = X_embed.shape[0]
		if n == 1:
			return [0]
		n_comp = min(12, max(2, n - 1))
		base_neighbors = 15
		n_neighbors = min(base_neighbors, max(5, n - 1), n - 1)
		reducer = self._make_umap(n_comp, n_neighbors, random_state)
		X_red = reducer.fit_transform(X_embed)
		labels = self._make_hdbscan().fit_predict(X_red)
		xp = self._xp
		labels_x = labels if hasattr(labels, "dtype") else xp.asarray(labels)
		valid_mask = labels_x != -1
		uniq_x = xp.unique(labels_x[valid_mask])
		if uniq_x.size == 0:
			return [0 for _ in range(len(X_embed))]
		sorted_uniq = xp.sort(uniq_x)
		mapped_x = xp.where(labels_x == -1, -1, xp.searchsorted(sorted_uniq, labels_x))
		k = int(sorted_uniq.size)
		centers = _compute_centers_from_labels(X_red, mapped_x, k, xp)
		centers /= (xp.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
		noise_idx = xp.where(mapped_x == -1)[0]
		if noise_idx.size > 0:
			dists = _pairwise_squared_euclidean(X_red[noise_idx], centers, xp)
			argmins = xp.argmin(dists, axis=1)
			mapped_x[noise_idx] = argmins
		mapped_np = mapped_x.get() if hasattr(mapped_x, "get") else np.asarray(mapped_x)
		return [int(i) for i in mapped_np]


 
