"""
fretnet_augmentation.py — FretNet学習用データ拡張モジュール
============================================================
HCQT特徴量テンソルおよび音声波形レベルでのデータ拡張を提供。
FretNetのGuitarSetPlus DatasetをラップしてOn-the-flyで拡張を適用。

Augmentations:
  - SpecAugment (Time Masking + Frequency Masking)
  - Frequency Axis Shift (Pitch Shift on HCQT features)
  - Gaussian Noise (on features)
  - Random Gain (on features)
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset

try:
    import amt_tools.tools as tools
except ImportError:
    tools = None


class HCQTAugmenter:
    """HCQT特徴量テンソルに対するデータ拡張クラス。
    
    FretNet用HCQT features の shape: (T, C, F, 1) or (T, C, F)
    where T=time frames, C=harmonic channels, F=frequency bins
    """
    
    def __init__(
        self,
        enable=True,
        # SpecAugment
        specaug_enabled=True,
        specaug_time_mask_param=30,
        specaug_freq_mask_param=15,
        specaug_num_time_masks=2,
        specaug_num_freq_masks=2,
        specaug_p=0.7,
        # Frequency axis shift (pitch shift on HCQT)
        freq_shift_enabled=True,
        freq_shift_max_bins=6,  # max bins to shift (each bin = 1/3 semitone for 36 bins/octave)
        freq_shift_p=0.4,
        # Gaussian noise
        noise_enabled=True,
        noise_std_range=(0.01, 0.05),
        noise_p=0.5,
        # Random gain
        gain_enabled=True,
        gain_range=(0.7, 1.3),
        gain_p=0.5,
    ):
        self.enable = enable
        
        # SpecAugment
        self.specaug_enabled = specaug_enabled
        self.specaug_p = specaug_p
        if specaug_enabled:
            self.time_masks = nn.ModuleList([
                torchaudio.transforms.TimeMasking(time_mask_param=specaug_time_mask_param)
                for _ in range(specaug_num_time_masks)
            ])
            self.freq_masks = nn.ModuleList([
                torchaudio.transforms.FrequencyMasking(freq_mask_param=specaug_freq_mask_param)
                for _ in range(specaug_num_freq_masks)
            ])
        
        # Frequency shift
        self.freq_shift_enabled = freq_shift_enabled
        self.freq_shift_max_bins = freq_shift_max_bins
        self.freq_shift_p = freq_shift_p
        
        # Noise
        self.noise_enabled = noise_enabled
        self.noise_std_range = noise_std_range
        self.noise_p = noise_p
        
        # Gain
        self.gain_enabled = gain_enabled
        self.gain_range = gain_range
        self.gain_p = gain_p
    
    def __call__(self, features, tablature=None):
        """Apply augmentations to HCQT features and optionally adjust tablature labels.
        
        Args:
            features: torch.Tensor of shape (T, C, F, 1) — HCQT features
            tablature: torch.Tensor of shape (6, T) — tablature labels, optional
            
        Returns:
            augmented_features, adjusted_tablature (or original tablature if not shifted)
        """
        if not self.enable:
            return features, tablature
        
        # numpy → torch 自動変換
        was_numpy = isinstance(features, np.ndarray)
        if was_numpy:
            features = torch.from_numpy(features).float()
        if tablature is not None and isinstance(tablature, np.ndarray):
            tablature = torch.from_numpy(tablature).long()
        
        had_4d = features.dim() == 4
        if had_4d:
            # (T, C, F, 1) -> (T, C, F)
            features = features.squeeze(-1)
        
        T, C, F = features.shape
        
        # 1. Frequency axis shift (must be done before SpecAugment)
        if self.freq_shift_enabled and random.random() < self.freq_shift_p:
            shift = random.randint(-self.freq_shift_max_bins, self.freq_shift_max_bins)
            if shift != 0:
                features = self._freq_shift(features, shift)
                if tablature is not None:
                    tablature = self._adjust_tablature_for_freq_shift(
                        tablature, shift, num_frets=19
                    )
        
        # 2. SpecAugment (applied per-channel)
        if self.specaug_enabled and random.random() < self.specaug_p:
            features = self._apply_specaugment(features)
        
        # 3. Gaussian noise
        if self.noise_enabled and random.random() < self.noise_p:
            noise_std = random.uniform(*self.noise_std_range)
            noise = torch.randn_like(features) * noise_std
            features = features + noise
        
        # 4. Random gain
        if self.gain_enabled and random.random() < self.gain_p:
            gain = random.uniform(*self.gain_range)
            features = features * gain
        
        if had_4d:
            features = features.unsqueeze(-1)
        
        return features, tablature
    
    def _freq_shift(self, features, shift_bins):
        """Shift features along frequency axis.
        
        Args:
            features: (T, C, F) tensor
            shift_bins: positive = shift up (higher pitch), negative = shift down
        """
        T, C, F = features.shape
        shifted = torch.zeros_like(features)
        
        if shift_bins > 0:
            # Shift up: higher frequency bins get filled
            shifted[:, :, shift_bins:] = features[:, :, :F - shift_bins]
        elif shift_bins < 0:
            # Shift down: lower frequency bins get filled
            abs_shift = abs(shift_bins)
            shifted[:, :, :F - abs_shift] = features[:, :, abs_shift:]
        else:
            shifted = features
        
        return shifted
    
    def _adjust_tablature_for_freq_shift(self, tablature, shift_bins, num_frets=19,
                                          bins_per_semitone=3):
        """Adjust tablature labels for frequency shift.
        
        Each frequency bin corresponds to 1/bins_per_semitone semitones.
        Shifting by N bins = shifting by N/bins_per_semitone semitones = N/bins_per_semitone frets.
        
        Args:
            tablature: (6, T) tensor
            shift_bins: number of bins shifted
            num_frets: max fret number
            bins_per_semitone: bins per semitone (36 bins/octave / 12 = 3)
        """
        semitone_shift = shift_bins / bins_per_semitone
        fret_shift = round(semitone_shift)
        
        if fret_shift == 0:
            return tablature
        
        adjusted = tablature.clone()
        # Only adjust active notes (non-negative values, -1 means silence)
        active_mask = adjusted >= 0
        adjusted[active_mask] = adjusted[active_mask] + fret_shift
        
        # Invalidate notes outside playable range
        invalid_mask = active_mask & ((adjusted < 0) | (adjusted > num_frets))
        adjusted[invalid_mask] = -1  # Mark as silence
        
        return adjusted
    
    def _apply_specaugment(self, features):
        """Apply SpecAugment to each harmonic channel.
        
        Input: (T, C, F) -> reshape for torchaudio -> (T, C, F)
        torchaudio expects (batch, freq, time) for FrequencyMasking
        and (batch, freq, time) for TimeMasking
        """
        T, C, F = features.shape
        
        for c in range(C):
            # (T, F) -> (1, F, T) for torchaudio
            channel = features[:, c, :].permute(1, 0).unsqueeze(0)  # (1, F, T)
            
            for mask in self.freq_masks:
                channel = mask(channel)
            for mask in self.time_masks:
                channel = mask(channel)
            
            # (1, F, T) -> (T, F)
            features[:, c, :] = channel.squeeze(0).permute(1, 0)
        
        return features


class AugmentedDatasetWrapper(Dataset):
    """Wraps a FretNet-compatible dataset and applies HCQT augmentation on-the-fly.
    
    Works with GuitarSetPlus datasets that return dicts with KEY_FEATS and KEY_TABLATURE.
    """
    
    def __init__(self, base_dataset, augmenter=None):
        """
        Args:
            base_dataset: Original FretNet dataset (GuitarSetPlus, SynthGuitarDataset, etc.)
            augmenter: HCQTAugmenter instance, or None to disable
        """
        self.base_dataset = base_dataset
        self.augmenter = augmenter or HCQTAugmenter(enable=False)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        if isinstance(sample, dict):
            features_key = tools.KEY_FEATS if tools else 'features'
            tablature_key = tools.KEY_TABLATURE if tools else 'tablature'
            
            features = sample.get(features_key)
            tablature = sample.get(tablature_key)
            
            if features is not None:
                aug_features, aug_tablature = self.augmenter(features, tablature)
                sample[features_key] = aug_features
                if aug_tablature is not None:
                    sample[tablature_key] = aug_tablature
        
        return sample
    
    # Proxy attributes from base dataset
    def __getattr__(self, name):
        if name in ('base_dataset', 'augmenter'):
            raise AttributeError(name)
        return getattr(self.base_dataset, name)
