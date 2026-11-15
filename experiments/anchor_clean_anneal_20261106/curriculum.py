"""Fixed 20-epoch anchor-dropout curriculum for an isolated experiment."""

from isaa.data.rtmw_zip_dataset import RTMWZipDataset


SCHEDULE = {
    "name": "anchor_p025_clean50_to_clean100_v1",
    "epochs": 20,
    "dropout_probability": 0.025,
    "clean_probability_first_ten": 0.5,
    "clean_probability_epoch_20": 1.0,
}


def clean_probability(epoch):
    if not 1 <= epoch <= 20:
        raise ValueError("Training epoch must be between 1 and 20")
    return 0.5 + 0.5 * max(0, epoch - 10) / 10


class CurriculumDataset(RTMWZipDataset):
    def _refresh_late_augmentation(self):
        if self.split != "train" or not self.augment:
            return
        epoch = int(self._epoch_state[0].item())
        if epoch == 0:
            return
        self.augmentation_config["precompression_anchor_clean_sample_probability"] = clean_probability(epoch)
