"""Slice raw observations before confidence conversion or temporal processing."""
import io
import numpy as np
import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _normalize_scores


class InputJointDataset(RTMWZipDataset):
    def __init__(self, *args, input_indices, **kwargs):
        self.selected_indices = tuple(input_indices)
        super().__init__(*args, num_joints=len(input_indices), feature_mode="raw", **kwargs)

    def _read_raw_sample(self, index):
        with np.load(io.BytesIO(self._get_archive().read(self.members[index])), allow_pickle=False) as data:
            points = np.asarray(data["keypoints"], dtype=np.float32)
            scores = np.asarray(data["scores"], dtype=np.float32)
        if points.ndim != 4 or points.shape[-1] != 2 or scores.shape != points.shape[:-1]:
            raise ValueError("Expected T x M x V x 2 keypoints and T x M x V scores")
        if points.shape[2] == 133:
            points = points[:, :self.max_persons, self.selected_indices, :]
            scores = scores[:, :self.max_persons, self.selected_indices]
        elif points.shape[2] == len(self.selected_indices):
            # Already compact files must use the manifest's exact node order.
            points = points[:, :self.max_persons]
            scores = scores[:, :self.max_persons]
        else:
            raise ValueError("ZIP nodes must be RTMW-133 or the manifest's compact node order")
        coordinate_valid = np.isfinite(points).all(-1)
        scores = _normalize_scores(scores, self.score_normalization)
        scores = np.where(coordinate_valid, scores, 0.).astype(np.float32)
        points = np.where(coordinate_valid[..., None], points, 0.).astype(np.float32)
        raw = torch.from_numpy(np.concatenate((points.transpose(3, 0, 2, 1), scores.transpose(0, 2, 1)[None]), axis=0))
        valid = torch.from_numpy(coordinate_valid.any(axis=(1, 2)))
        if raw.size(3) < self.max_persons:
            raw = torch.cat((raw, raw.new_zeros(3, raw.size(1), raw.size(2), self.max_persons - raw.size(3))), dim=3)
        return raw.contiguous(), valid.bool().contiguous()
