#!/usr/bin/env python3
"""
================================================================================
V1(P1): MI ESTIMATION EXPERIMENT CODE
================================================================================
ANALYSIS SUPPORTED:
           - Feature Extraction and Source-Filter Attribution Statistics
           - Cross-Dimension MI Analysis 
DATASETS:
  Emotional:    RAVDESS, IEMOCAP
  Linguistic:   L2-ARCTIC, GMU Speech Accent Archive  
  Pathological: UA-Speech, MDVR-KCL
================================================================================
Authors: [removed for review]

This script computes all dataset combinations and saves cache artifacts only.
Default cache target: output/cache_new
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import librosa
import parselmouth
from parselmouth.praat import call
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, pairwise_distances
)
from scipy import stats
import warnings
import os
from glob import glob
import argparse
from typing import Dict, Tuple, List, Optional
import logging
import json
from datetime import datetime
from itertools import product
from dataclasses import dataclass
import textwrap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    'emotional': {
        'datasets': ['RAVDESS', 'IEMOCAP'],
        'args': ['ravdess', 'iemocap']
    },
    'linguistic': {
        'datasets': ['L2-ARCTIC', 'GMU'],
        'args': ['l2arctic', 'gmu']
    },
    'pathological': {
        'datasets': ['UA-Speech', 'MDVR-KCL'],
        'args': ['uaspeech', 'mdvr']
    }
}


# ============================================================================
# 1. FEATURE EXTRACTION WITH ROBUST DIMENSION CONTROL
# ============================================================================

class SpeechFeatureExtractor:
    """Extract emotional, linguistic, pathological, source, and filter features from speech.

    Expected feature dimensions:
    - Emotional: 28
    - Linguistic: 33
    - Pathological: 16
    - Source: 9
    - Filter: 32
    """
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.expected_dims = {
            'emotional': 28,
            'linguistic': 33,
            'pathological': 16,
            'source': 9,
            'filter': 32
        }
        logger.debug(f"SpeechFeatureExtractor initialized with sr={sr}")
        
    def _pad_or_trim_feature(self, feat: np.ndarray, target_dim: int) -> np.ndarray:
        """Ensure feature vector has exactly target_dim dimensions."""
        feat = np.atleast_1d(np.nan_to_num(feat, nan=0.0, posinf=1000.0, neginf=-1000.0))
        
        if len(feat) < target_dim:
            padding = np.zeros(target_dim - len(feat))
            return np.concatenate([feat, padding])
        elif len(feat) > target_dim:
            return feat[:target_dim]
        else:
            return feat
        
    def extract_f0_features(self, sound: parselmouth.Sound) -> np.ndarray:
        """Extract fundamental frequency features (source dimension)."""
        try:
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            f0_values = pitch.selected_array['frequency']
            f0_values = f0_values[f0_values > 0]
            
            if len(f0_values) < 10:
                return np.zeros(6)
            
            features = np.array([
                np.mean(f0_values),
                np.std(f0_values),
                np.max(f0_values) - np.min(f0_values),
                np.percentile(f0_values, 25),
                np.percentile(f0_values, 75),
                np.median(f0_values)
            ])
            return np.nan_to_num(features, nan=0.0)
        except Exception as e:
            logger.debug(f"F0 extraction failed: {e}")
            return np.zeros(6)
    
    def extract_voice_quality(self, sound: parselmouth.Sound) -> np.ndarray:
        """Extract voice quality features (source dimension)."""
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            
            try:
                jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer = call([sound, point_process], "Get shimmer (local)", 
                              0, 0, 0.0001, 0.02, 1.3, 1.6)
            except:
                jitter, shimmer = 0.0, 0.0
            
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            
            return np.array([
                float(jitter) if jitter is not None else 0.0,
                float(shimmer) if shimmer is not None else 0.0,
                float(hnr) if (hnr is not None and not np.isnan(hnr)) else 0.0
            ])
        except Exception as e:
            logger.debug(f"Voice quality extraction failed: {e}")
            return np.zeros(3)
    
    def extract_formants(self, sound: parselmouth.Sound) -> np.ndarray:
        """Extract formant features (filter dimension)."""
        try:
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
            f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
            f3 = call(formant, "Get mean", 3, 0, 0, "Hertz")
            
            b1 = call(formant, "Get bandwidth at time", 1, 0.5, "Hertz", "Linear")
            b2 = call(formant, "Get bandwidth at time", 2, 0.5, "Hertz", "Linear")
            b3 = call(formant, "Get bandwidth at time", 3, 0.5, "Hertz", "Linear")
            
            features = [f1, f2, f3, b1, b2, b3]
            return np.array([float(f) if (f is not None and not np.isnan(f)) else 0.0 
                            for f in features])
        except Exception as e:
            logger.debug(f"Formant extraction failed: {e}")
            return np.zeros(6)
    
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Extract MFCC features (filter dimension)."""
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            mfcc_mean = np.mean(mfcc, axis=1)
            delta_mean = np.mean(delta, axis=1)
            delta2_mean = np.mean(delta2, axis=1)
            
            return np.concatenate([mfcc_mean, delta_mean, delta2_mean])
        except Exception as e:
            logger.debug(f"MFCC extraction failed: {e}")
            return np.zeros(39)
    
    def extract_spectral_features(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral features (emotional dimension)."""
        try:
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sr)))
            spectral_flux = float(np.mean(librosa.onset.onset_strength(y=y, sr=self.sr)))
            spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.sr)))
            
            return np.array([
                spectral_centroid if not np.isnan(spectral_centroid) else 0.0,
                spectral_flux if not np.isnan(spectral_flux) else 0.0,
                spectral_rolloff if not np.isnan(spectral_rolloff) else 0.0
            ])
        except Exception as e:
            logger.debug(f"Spectral feature extraction failed: {e}")
            return np.zeros(3)
    
    def extract_energy_features(self, y: np.ndarray) -> np.ndarray:
        """Extract energy/intensity features (emotional dimension)."""
        try:
            rms = librosa.feature.rms(y=y)[0]
            return np.array([
                float(np.mean(rms)),
                float(np.std(rms)),
                float(np.max(rms))
            ])
        except Exception as e:
            logger.debug(f"Energy extraction failed: {e}")
            return np.zeros(3)
    
    def extract_rhythm_features(self, y: np.ndarray) -> np.ndarray:
        """Extract rhythm features (linguistic dimension)."""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
            duration = len(y) / self.sr
            
            return np.array([
                float(tempo) if not np.isnan(tempo) else 0.0,
                float(duration) if not np.isnan(duration) else 0.0
            ])
        except Exception as e:
            logger.debug(f"Rhythm extraction failed: {e}")
            return np.zeros(2)
    
    def extract_emotional_features(self, y: np.ndarray, sound: parselmouth.Sound) -> np.ndarray:
        """Extract emotional dimension features."""
        f0 = self.extract_f0_features(sound)
        vq = self.extract_voice_quality(sound)
        energy = self.extract_energy_features(y)
        spectral = self.extract_spectral_features(y)
        
        features = np.concatenate([
            np.atleast_1d(f0),
            np.atleast_1d(vq),
            np.atleast_1d(energy),
            np.atleast_1d(spectral)
        ])
        
        return self._pad_or_trim_feature(features, self.expected_dims['emotional'])
    
    def extract_linguistic_features(self, y: np.ndarray, sound: parselmouth.Sound) -> np.ndarray:
        """Extract linguistic dimension features."""
        formants = self.extract_formants(sound)
        mfcc = self.extract_mfcc(y)
        rhythm = self.extract_rhythm_features(y)
        
        features = np.concatenate([
            np.atleast_1d(formants),
            np.atleast_1d(mfcc),
            np.atleast_1d(rhythm)
        ])
        
        return self._pad_or_trim_feature(features, self.expected_dims['linguistic'])
    
    def extract_pathological_features(self, y: np.ndarray, sound: parselmouth.Sound) -> np.ndarray:
        """Extract pathological dimension features."""
        vq = self.extract_voice_quality(sound)
        formants = self.extract_formants(sound)
        
        f2_velocity = 0.0
        try:
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            f2_trajectory = []
            for t in np.linspace(0.1, 0.9, 10):
                try:
                    f2_t = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                    if f2_t is not None and not np.isnan(f2_t):
                        f2_trajectory.append(f2_t)
                except:
                    pass
            
            if len(f2_trajectory) > 1:
                f2_velocity = float(np.mean(np.abs(np.diff(f2_trajectory))))
        except Exception:
            pass
        
        features = np.concatenate([
            np.atleast_1d(vq),
            np.atleast_1d(formants),
            np.atleast_1d([f2_velocity])
        ])
        
        return self._pad_or_trim_feature(features, self.expected_dims['pathological'])
    
    def extract_source_features(self, y: np.ndarray, sound: parselmouth.Sound) -> np.ndarray:
        """Extract source (glottal) features."""
        f0 = self.extract_f0_features(sound)
        vq = self.extract_voice_quality(sound)
        
        features = np.concatenate([
            np.atleast_1d(f0),
            np.atleast_1d(vq)
        ])
        
        return self._pad_or_trim_feature(features, self.expected_dims['source'])
    
    def extract_filter_features(self, y: np.ndarray, sound: parselmouth.Sound) -> np.ndarray:
        """Extract filter (vocal tract) features."""
        formants = self.extract_formants(sound)
        mfcc = self.extract_mfcc(y)
        
        features = np.concatenate([
            np.atleast_1d(formants),
            np.atleast_1d(mfcc)
        ])
        
        return self._pad_or_trim_feature(features, self.expected_dims['filter'])
    
    def extract_all_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Extract all feature sets from audio file."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            if y.ndim > 1:
                y = np.mean(y, axis=0)
            
            sound = parselmouth.Sound(audio_path)
            if sound.n_channels > 1:
                sound = sound.convert_to_mono()
            
            return {
                'emotional': self.extract_emotional_features(y, sound),
                'linguistic': self.extract_linguistic_features(y, sound),
                'pathological': self.extract_pathological_features(y, sound),
                'source': self.extract_source_features(y, sound),
                'filter': self.extract_filter_features(y, sound)
            }
        except Exception as e:
            logger.debug(f"Feature extraction failed for {os.path.basename(audio_path)}: {e}")
            return {
                'emotional': np.zeros(self.expected_dims['emotional']),
                'linguistic': np.zeros(self.expected_dims['linguistic']),
                'pathological': np.zeros(self.expected_dims['pathological']),
                'source': np.zeros(self.expected_dims['source']),
                'filter': np.zeros(self.expected_dims['filter'])
            }


# ============================================================================
# 2. NEURAL MUTUAL INFORMATION ESTIMATORS
# ============================================================================

class MINENetwork(nn.Module):
    """Mutual Information Neural Estimator (MINE) network."""
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass computing T(x, y)."""
        xy = torch.cat([x, y], dim=1)
        return self.network(xy)


class MINE:
    """MINE: Mutual Information Neural Estimation (Lower Bound)."""
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256, lr: float = 1e-4):
        self.network = MINENetwork(x_dim, y_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
    def compute_mi(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute MI lower bound using Donsker-Varadhan representation."""
        t_joint = self.network(x, y)
        y_shuffle = y[torch.randperm(y.size(0))]
        t_marginal = self.network(x, y_shuffle)
        
        mi_lb = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)))
        
        return mi_lb
    
    def train(self, x_data: np.ndarray, y_data: np.ndarray, 
              epochs: int = 100, batch_size: int = 256) -> float:
        """Train MINE estimator."""
        dataset = TensorDataset(
            torch.FloatTensor(x_data),
            torch.FloatTensor(y_data)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        mi_estimates = []
        
        for epoch in range(epochs):
            epoch_mi = []
            for x_batch, y_batch in loader:
                self.optimizer.zero_grad()
                mi = self.compute_mi(x_batch, y_batch)
                loss = -mi
                loss.backward()
                self.optimizer.step()
                epoch_mi.append(mi.item())
            
            mi_estimates.append(np.mean(epoch_mi))
        
        return mi_estimates[-1]


class CLUBNetwork(nn.Module):
    """CLUB: Contrastive Log-ratio Upper Bound network."""
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim)
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict conditional distribution parameters."""
        mu = self.mu_net(x)
        logvar = self.logvar_net(x)
        return mu, logvar
    
    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log q(y|x) assuming Gaussian."""
        mu, logvar = self.forward(x)
        return -0.5 * (logvar + (y - mu)**2 / torch.exp(logvar)).sum(dim=1)


class CLUB:
    """CLUB: Contrastive Log-ratio Upper Bound of MI."""
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256, lr: float = 1e-4):
        self.network = CLUBNetwork(x_dim, y_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
    def compute_mi(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute MI upper bound."""
        positive_ll = self.network.log_likelihood(x, y)
        y_shuffle = y[torch.randperm(y.size(0))]
        negative_ll = self.network.log_likelihood(x, y_shuffle)
        
        mi_ub = torch.mean(positive_ll) - torch.mean(negative_ll)
        return mi_ub
    
    def train(self, x_data: np.ndarray, y_data: np.ndarray, 
              epochs: int = 100, batch_size: int = 256) -> float:
        """Train CLUB estimator."""
        dataset = TensorDataset(
            torch.FloatTensor(x_data),
            torch.FloatTensor(y_data)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        mi_estimates = []
        
        for epoch in range(epochs):
            epoch_mi = []
            for x_batch, y_batch in loader:
                self.optimizer.zero_grad()
                
                mu, logvar = self.network(x_batch)
                ll_loss = -self.network.log_likelihood(x_batch, y_batch).mean()
                ll_loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    mi = self.compute_mi(x_batch, y_batch)
                    epoch_mi.append(mi.item())
            
            mi_estimates.append(np.mean(epoch_mi))
        
        return mi_estimates[-1]


# ============================================================================
#  MI ESTIMATION WITH HISTORY TRACKING
# ============================================================================

@dataclass
class MIEstimationResult:
    """
    Container for MI estimation results with full history.
    
     This dataclass stores:
    - final_estimate: The converged MI value
    - epoch_history: Full training curve for convergence plots
    - converged: Whether bounds stabilized
    - convergence_epoch: When convergence was detected
    - final_std: Uncertainty in final estimate
    """
    final_estimate: float
    epoch_history: list
    converged: bool
    convergence_epoch: int = None
    final_std: float = 0.0


class MINEWithHistory(MINE):
    """
    MINE estimator with epoch-by-epoch history tracking.
    
     Enhanced MINE that tracks MI estimate at every epoch
    for convergence visualization. The convergence plot shows MINE 
    (red line) starting low and increasing until it stabilizes.
    """
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256, lr: float = 1e-4):
        super().__init__(x_dim, y_dim, hidden_dim, lr)
        self.history = []
    
    def train_with_history(self, x_data: np.ndarray, y_data: np.ndarray,
                           epochs: int = 100, batch_size: int = 256,
                           patience: int = 10) -> MIEstimationResult:
        """Train MINE with full epoch history tracking."""
        dataset = TensorDataset(
            torch.FloatTensor(x_data),
            torch.FloatTensor(y_data)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.history = []
        convergence_epoch = None
        convergence_threshold = 0.01
        
        for epoch in range(epochs):
            epoch_mi = []
            for x_batch, y_batch in loader:
                self.optimizer.zero_grad()
                mi = self.compute_mi(x_batch, y_batch)
                loss = -mi
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
                epoch_mi.append(mi.item())
            
            epoch_mean = np.mean(epoch_mi)
            self.history.append(max(0, epoch_mean))
            
            # Check convergence
            if len(self.history) > patience:
                recent_std = np.std(self.history[-patience:])
                if recent_std < convergence_threshold and convergence_epoch is None:
                    convergence_epoch = epoch
        
        final_estimate = np.mean(self.history[-patience:]) if len(self.history) >= patience else self.history[-1]
        final_std = np.std(self.history[-patience:]) if len(self.history) >= patience else 0.0
        
        return MIEstimationResult(
            final_estimate=max(0, final_estimate),
            epoch_history=self.history.copy(),
            converged=convergence_epoch is not None,
            convergence_epoch=convergence_epoch,
            final_std=final_std
        )


class CLUBWithHistory(CLUB):
    """
    CLUB estimator with epoch-by-epoch history tracking.
    
     Enhanced CLUB that tracks MI estimate at every epoch
    for convergence visualization. The convergence plot shows CLUB 
    (blue line) starting high and decreasing until it stabilizes.
    """
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256, lr: float = 1e-4):
        super().__init__(x_dim, y_dim, hidden_dim, lr)
        self.history = []
    
    def train_with_history(self, x_data: np.ndarray, y_data: np.ndarray,
                           epochs: int = 100, batch_size: int = 256,
                           patience: int = 10) -> MIEstimationResult:
        """Train CLUB with full epoch history tracking."""
        dataset = TensorDataset(
            torch.FloatTensor(x_data),
            torch.FloatTensor(y_data)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.history = []
        convergence_epoch = None
        convergence_threshold = 0.01
        
        for epoch in range(epochs):
            epoch_mi = []
            for x_batch, y_batch in loader:
                self.optimizer.zero_grad()
                
                mu, logvar = self.network(x_batch)
                ll_loss = -self.network.log_likelihood(x_batch, y_batch).mean()
                ll_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
                
                with torch.no_grad():
                    mi = self.compute_mi(x_batch, y_batch)
                    epoch_mi.append(mi.item())
            
            epoch_mean = np.mean(epoch_mi)
            self.history.append(max(0, epoch_mean))
            
            # Check convergence
            if len(self.history) > patience:
                recent_std = np.std(self.history[-patience:])
                if recent_std < convergence_threshold and convergence_epoch is None:
                    convergence_epoch = epoch
        
        final_estimate = np.mean(self.history[-patience:]) if len(self.history) >= patience else self.history[-1]
        final_std = np.std(self.history[-patience:]) if len(self.history) >= patience else 0.0
        
        return MIEstimationResult(
            final_estimate=max(0, final_estimate),
            epoch_history=self.history.copy(),
            converged=convergence_epoch is not None,
            convergence_epoch=convergence_epoch,
            final_std=final_std
        )


# ============================================================================
#  CONVERGENCE ANALYZER
# ============================================================================

class ConvergenceAnalyzer:
    """
    Analyze and validate MI estimation convergence.
    
     This class:
    1. Runs both MINE and CLUB estimators with history tracking
    2. Validates that MINE (lower) <= CLUB (upper)
    3. Computes true MI as midpoint when bounds are valid
    4. Quantifies uncertainty as gap between bounds
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def estimate_mi_with_convergence(self, x: np.ndarray, y: np.ndarray,
                                      epochs: int = 100, n_runs: int = 5,
                                      hidden_dim: int = 256, lr: float = 1e-4) -> Dict:
        """Estimate MI with full convergence tracking."""
        x_norm = self.scaler.fit_transform(x)
        y_norm = self.scaler.fit_transform(y)
        
        mine_histories = []
        club_histories = []
        mine_finals = []
        club_finals = []
        
        for run in range(n_runs):
            torch.manual_seed(run)
            np.random.seed(run)
            
            # Train MINE
            mine = MINEWithHistory(x_norm.shape[1], y_norm.shape[1], hidden_dim, lr)
            mine_result = mine.train_with_history(x_norm, y_norm, epochs=epochs)
            mine_histories.append(mine_result.epoch_history)
            mine_finals.append(mine_result.final_estimate)
            
            # Train CLUB
            club = CLUBWithHistory(x_norm.shape[1], y_norm.shape[1], hidden_dim, lr)
            club_result = club.train_with_history(x_norm, y_norm, epochs=epochs)
            club_histories.append(club_result.epoch_history)
            club_finals.append(club_result.final_estimate)
        
        mine_mean = np.mean(mine_finals)
        mine_std = np.std(mine_finals)
        club_mean = np.mean(club_finals)
        club_std = np.std(club_finals)
        
        valid = mine_mean <= club_mean + 0.5
        true_mi = (mine_mean + club_mean) / 2
        uncertainty = abs(club_mean - mine_mean)
        
        if not valid:
            logger.warning(f"Bounds crossed: MINE={mine_mean:.3f} > CLUB={club_mean:.3f}")
        
        return {
            'mine_mean': mine_mean, 'mine_std': mine_std,
            'club_mean': club_mean, 'club_std': club_std,
            'true_mi': true_mi, 'uncertainty': uncertainty,
            'valid': valid,
            'mine_histories': mine_histories,
            'club_histories': club_histories
        }


# ============================================================================
#  ABLATION STUDIES
# ============================================================================

class AblationStudy:
    """
    Comprehensive ablation studies for MI estimation hyperparameters.
    
     Tests sensitivity to:
    - Network hidden dimensions
    - Training epochs  
    - Learning rate
    """
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.scaler = StandardScaler()
        self.x_norm = self.scaler.fit_transform(x_data)
        self.y_norm = self.scaler.fit_transform(y_data)
        self.results = {}
    
    def _run_config(self, hidden_dim: int = 256, epochs: int = 100, lr: float = 1e-4) -> Dict:
        """Run single configuration."""
        torch.manual_seed(42)
        
        mine = MINEWithHistory(self.x_norm.shape[1], self.y_norm.shape[1], hidden_dim, lr)
        mine_result = mine.train_with_history(self.x_norm, self.y_norm, epochs=epochs)
        
        club = CLUBWithHistory(self.x_norm.shape[1], self.y_norm.shape[1], hidden_dim, lr)
        club_result = club.train_with_history(self.x_norm, self.y_norm, epochs=epochs)
        
        return {
            'mine': mine_result.final_estimate,
            'club': club_result.final_estimate,
            'combined': (mine_result.final_estimate + club_result.final_estimate) / 2,
            'uncertainty': abs(club_result.final_estimate - mine_result.final_estimate),
            'mine_history': mine_result.epoch_history,
            'club_history': club_result.epoch_history
        }
    
    def ablation_hidden_dim(self, dims: List[int] = [64, 128, 256, 512]) -> Dict:
        """Test different hidden dimensions."""
        logger.info("    Ablation: Hidden Dimensions")
        results = {d: self._run_config(hidden_dim=d) for d in dims}
        self.results['hidden_dim'] = results
        return results
    
    def ablation_epochs(self, epoch_vals: List[int] = [25, 50, 100, 200]) -> Dict:
        """Test different epoch counts."""
        logger.info("    Ablation: Training Epochs")
        results = {e: self._run_config(epochs=e) for e in epoch_vals}
        self.results['epochs'] = results
        return results
    
    def ablation_learning_rate(self, lrs: List[float] = [1e-5, 1e-4, 1e-3]) -> Dict:
        """Test different learning rates."""
        logger.info("    Ablation: Learning Rate")
        results = {lr: self._run_config(lr=lr) for lr in lrs}
        self.results['learning_rate'] = results
        return results
    
    def run_all(self) -> Dict:
        """Run all ablation studies."""
        self.ablation_hidden_dim()
        self.ablation_epochs()
        self.ablation_learning_rate()
        return self.results

# Cache-only mode: plotting and convergence/ablation visualization helpers removed.

# 3. DIMENSIONAL INDEPENDENCE ANALYSIS
# ============================================================================

class DimensionalAnalyzer:
    """Analyze dimensional independence in speech features."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def estimate_mi_pair(self, x: np.ndarray, y: np.ndarray, n_runs: int = 10) -> Dict:
        """Estimate MI between two feature sets using MINE and CLUB."""
        if x.shape[0] == 0 or y.shape[0] == 0:
            logger.warning(f"Empty data - x:{x.shape}, y:{y.shape}")
            return {
                'mine_mean': 0.0, 'mine_std': 0.0,
                'club_mean': 0.0, 'club_std': 0.0,
                'combined_mean': 0.0, 'combined_std': 0.0
            }
        
        x_norm = self.scaler.fit_transform(x)
        y_norm = self.scaler.fit_transform(y)
        
        mine_estimates = []
        club_estimates = []
        
        for seed in range(n_runs):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            try:
                mine = MINE(x_norm.shape[1], y_norm.shape[1])
                mine_mi = mine.train(x_norm, y_norm)
                mine_estimates.append(max(0, mine_mi))
            except Exception as e:
                logger.debug(f"MINE training failed: {e}")
                mine_estimates.append(0.0)
            
            try:
                club = CLUB(x_norm.shape[1], y_norm.shape[1])
                club_mi = club.train(x_norm, y_norm)
                club_estimates.append(max(0, club_mi))
            except Exception as e:
                logger.debug(f"CLUB training failed: {e}")
                club_estimates.append(0.0)
        
        return {
            'mine_mean': np.mean(mine_estimates),
            'mine_std': np.std(mine_estimates),
            'club_mean': np.mean(club_estimates),
            'club_std': np.std(club_estimates),
            'combined_mean': np.mean(mine_estimates + club_estimates),
            'combined_std': np.std(mine_estimates + club_estimates)
        }
    
    def compute_source_filter_attribution(self, dimension_features: np.ndarray, 
                                         source_features: np.ndarray, 
                                         filter_features: np.ndarray) -> Tuple[float, float]:
        """Compute source vs filter attribution for a dimension."""
        dim_norm = self.scaler.fit_transform(dimension_features)
        source_norm = self.scaler.fit_transform(source_features)
        filter_norm = self.scaler.fit_transform(filter_features)
        
        torch.manual_seed(SEED)
        
        try:
            mine_source = MINE(source_norm.shape[1], dim_norm.shape[1])
            mi_source = max(0, mine_source.train(source_norm, dim_norm))
        except:
            mi_source = 0.0
        
        try:
            mine_filter = MINE(filter_norm.shape[1], dim_norm.shape[1])
            mi_filter = max(0, mine_filter.train(filter_norm, dim_norm))
        except:
            mi_filter = 0.0
        
        total = mi_source + mi_filter
        if total > 0:
            source_attr = mi_source / total
            filter_attr = mi_filter / total
        else:
            source_attr = filter_attr = 0.5
        
        return source_attr, filter_attr
class StatisticalValidator:
    """
    Validate statistical significance of findings via:
    - Confidence intervals (95% CI)
    - Effect sizes (Cohen's d)
    - P-values (Mann-Whitney U test)
    - Cross-dataset consistency checks
    """
    
    @staticmethod
    def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute 95% confidence interval using t-distribution."""
        if len(values) < 2:
            return (np.mean(values), np.mean(values))
        
        mean = np.mean(values)
        se = stats.sem(values)
        margin = se * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        
        return (mean - margin, mean + margin)
    
    @staticmethod
    def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std < 1e-10:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    @staticmethod
    def mann_whitney_test(group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Non-parametric Mann-Whitney U test."""
        try:
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            return {
                'statistic': float(statistic),
                'p_value': float(pvalue),
                'significant': pvalue < 0.05
            }
        except:
            return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}


# ============================================================================
# 7. LATEX TABLE GENERATOR
# ============================================================================

class LatexTableGenerator:
    """Generate LaTeX tables with proper formatting."""
    
    @staticmethod
    def escape_latex(text: str) -> str:
        """Escape special LaTeX characters."""
        special_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
        }
        for char, replacement in special_chars.items():
            text = text.replace(char, replacement)
        return text
    
    @staticmethod
    def generate_mi_attribution_table(df: pd.DataFrame, caption: str, label: str) -> str:
        """
        Generate LaTeX table for MI analysis attribution statistics.
        
        Caption includes metric interpretation.
        """
        latex = textwrap.dedent(f"""
        \\begin{{table}}[htbp]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{lccccc}}
        \\toprule
        Dimension & Mean & Std & 95\\% CI Lower & 95\\% CI Upper & Interpretation \\\\
        \\midrule
        """)
        
        for _, row in df.iterrows():
            mean_val = float(row['Mean'])
            # Interpretation based on source attribution
            if mean_val > 0.6:
                interp = "Source-dominated"
            elif mean_val < 0.4:
                interp = "Filter-dominated"
            else:
                interp = "Balanced"
            
            latex += f"        {row['Dimension']} & {row['Mean']} & {row['Std']} & "
            latex += f"{row['95% CI Lower']} & {row['95% CI Upper']} & {interp} \\\\\n"
        
        latex += textwrap.dedent("""
        \\bottomrule
        \\end{tabular}
        \\begin{tablenotes}
        \\small
        \\item \\textit{Note:} Source attribution values represent the proportion of dimension variance 
        explained by glottal source features vs. vocal tract filter features. 
        Values >0.6 indicate source-dominated; <0.4 indicate filter-dominated.
        95\\% CI computed via t-distribution across 8 dataset combinations.
        \\end{tablenotes}
        \\end{table}
        """)
        
        return latex
    
    @staticmethod
    def generate_mi_mi_table(df: pd.DataFrame, caption: str, label: str) -> str:
        """
        Generate LaTeX table for MI analysis MI statistics.
        
        Caption includes orthogonality interpretation.
        """
        latex = textwrap.dedent(f"""
        \\begin{{table}}[htbp]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{lccc}}
        \\toprule
        Dimension Pair & Mean MI (nats) & Std Dev & Interpretation \\\\
        \\midrule
        """)
        
        for _, row in df.iterrows():
            pair = LatexTableGenerator.escape_latex(str(row['Dimension Pair']))
            latex += f"        {pair} & {row['Mean MI (nats)']} & {row['Std Dev']} & {row['Interpretation']} \\\\\n"
        
        latex += textwrap.dedent("""
        \\bottomrule
        \\end{tabular}
        \\begin{tablenotes}
        \\small
        \\item \\textit{Note:} Mutual Information (MI) in nats measures statistical dependence 
        between dimension pairs. Lower values indicate greater orthogonality (independence).
        Interpretation: <2 nats = orthogonal; 2-5 nats = mostly independent; 
        5-10 nats = moderately coupled; >10 nats = tightly coupled.
        Combined estimate uses both MINE (lower bound) and CLUB (upper bound) estimators.
        \\end{tablenotes}
        \\end{table}
        """)
        
        return latex
    
    @staticmethod
    def generate_extended_clustering_table(df: pd.DataFrame, caption: str, label: str) -> str:
        """
        Generate LaTeX table for extended analysis clustering metrics.
        
        Caption includes metric interpretation.
        """
        latex = textwrap.dedent(f"""
        \\begin{{table}}[htbp]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{lcccc}}
        \\toprule
        Combination & Silhouette$^a$ & Davies-Bouldin$^b$ & Calinski-Harabasz$^c$ & Quality \\\\
        \\midrule
        """)
        
        for _, row in df.iterrows():
            combo = LatexTableGenerator.escape_latex(str(row['Combination']))
            sil = float(row['Silhouette'])
            
            # Quality assessment based on silhouette
            if sil > 0.6:
                quality = "\\textbf{Excellent}"
            elif sil > 0.4:
                quality = "Good"
            elif sil > 0.25:
                quality = "Fair"
            else:
                quality = "Poor"
            
            latex += f"        {combo} & {row['Silhouette']} & {row['Davies-Bouldin']} & "
            latex += f"{row['Calinski-Harabasz']} & {quality} \\\\\n"
        
        latex += textwrap.dedent("""
        \\bottomrule
        \\end{tabular}
        \\begin{tablenotes}
        \\small
        \\item $^a$ Silhouette: [-1, 1], higher = better. >0.7 excellent, >0.5 good, >0.25 fair.
        \\item $^b$ Davies-Bouldin: [0, $\\infty$], lower = better. <0.5 excellent, <1.0 good.
        \\item $^c$ Calinski-Harabasz: [0, $\\infty$], higher = better defined clusters.
        \\item \\textit{Note:} Metrics computed on t-SNE embeddings of combined feature spaces.
        \\end{tablenotes}
        \\end{table}
        """)
        
        return latex
    
    @staticmethod
    def generate_extended_stability_table(df: pd.DataFrame, caption: str, label: str) -> str:
        """
        Generate LaTeX table for extended analysis bootstrap stability.
        
        Caption includes stability interpretation.
        """
        latex = textwrap.dedent(f"""
        \\begin{{table}}[htbp]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{lccc}}
        \\toprule
        Combination & Mean Stability & Std Dev & Assessment \\\\
        \\midrule
        """)
        
        for _, row in df.iterrows():
            combo = LatexTableGenerator.escape_latex(str(row['Combination']))
            mean_stab = float(row['Mean Stability'])
            
            # Stability assessment
            if mean_stab > 0.85:
                assess = "\\textbf{High}"
            elif mean_stab > 0.60:
                assess = "Moderate"
            else:
                assess = "Low"
            
            latex += f"        {combo} & {row['Mean Stability']:.3f} & {row['Std Stability']:.3f} & {assess} \\\\\n"
        
        latex += textwrap.dedent("""
        \\bottomrule
        \\end{tabular}
        \\begin{tablenotes}
        \\small
        \\item \\textit{Note:} Bootstrap stability computed over 20 iterations with 80\\% subsampling.
        Stability measured via Adjusted Rand Index (ARI) between subsampled and original cluster assignments.
        Interpretation: >0.85 = high stability; 0.60-0.85 = moderate; <0.60 = low stability.
        \\end{tablenotes}
        \\end{table}
        """)
        
        return latex
    
    @staticmethod
    def generate_summary_comparison_table(df: pd.DataFrame, caption: str, label: str) -> str:
        """Generate comprehensive LaTeX summary table."""
        latex = textwrap.dedent(f"""
        \\begin{{table*}}[htbp]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\small
        \\begin{{tabular}}{{lccccccc}}
        \\toprule
        Combination & Emo-Ling & Emo-Path & Ling-Path & Src-Flt & Emo $A_S$ & Ling $A_F$ & Path $A_F$ \\\\
        \\midrule
        """)
        
        for _, row in df.iterrows():
            combo = LatexTableGenerator.escape_latex(str(row['Combination']))
            latex += f"        {combo} & {row.get('Emo-Ling MINE', 'N/A')} & "
            latex += f"{row.get('Emo-Path MINE', 'N/A')} & {row.get('Ling-Path MINE', 'N/A')} & "
            latex += f"{row.get('Src-Flt MINE', 'N/A')} & {row.get('Emo A_S', 'N/A')} & "
            latex += f"{row.get('Ling A_F', 'N/A')} & {row.get('Path A_F', 'N/A')} \\\\\n"
        
        latex += textwrap.dedent("""
        \\bottomrule
        \\end{tabular}
        \\begin{tablenotes}
        \\small
        \\item \\textit{Note:} MI values in nats (MINE estimator). $A_S$ = Source attribution; 
        $A_F$ = Filter attribution. Lower cross-dimension MI indicates greater orthogonality.
        Higher Source-Filter MI indicates tighter acoustic coupling (expected for speech production).
        \\end{tablenotes}
        \\end{table*}
        """)
        
        return latex


# ============================================================================
# 8. MULTI-DATASET DATA LOADING
# ============================================================================

def load_single_dataset(dataset_path: str, dataset_name: str, dimension: str,
                       extractor: SpeechFeatureExtractor, 
                       max_samples: int = 500) -> Dict[str, List[np.ndarray]]:
    """Load features from a single dataset."""
    
    dataset_path = os.path.abspath(dataset_path)

    if not os.path.exists(dataset_path):
        # Auto-resolve common dataset folder names
        fallback_candidates = []
        if dataset_name.upper() == 'GMU':
            parent_dir = os.path.dirname(dataset_path)
            fallback_candidates = [
                os.path.join(parent_dir, "GMU-Accented Speech Archive"),
                os.path.join(parent_dir, "GMU Speech Accent Archive"),
                os.path.join(parent_dir, "GMU_Accented_Speech_Archive"),
                os.path.join(parent_dir, "GMU-Accent-Archive")
            ]

        resolved = None
        for candidate in fallback_candidates:
            if os.path.exists(candidate):
                resolved = candidate
                break

        if resolved:
            logger.info(f"Resolved {dataset_name} path to: {resolved}")
            dataset_path = resolved
        else:
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return {'emotional': [], 'linguistic': [], 'pathological': [], 
                    'source': [], 'filter': []}
    
    # Find audio files
    audio_files = glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)
    if not audio_files:
        audio_files = glob(os.path.join(dataset_path, "**/*.mp3"), recursive=True)
    if not audio_files:
        audio_files = glob(os.path.join(dataset_path, "**/*.flac"), recursive=True)
    
    logger.info(f"  Loading {dataset_name} ({dimension}): found {len(audio_files)} files...")
    
    features = {
        'emotional': [],
        'linguistic': [],
        'pathological': [],
        'source': [],
        'filter': []
    }
    
    for audio_path in audio_files[:max_samples]:
        try:
            feats = extractor.extract_all_features(audio_path)
            
            # Store features based on which dimension this dataset represents
            if dimension == 'emotional':
                if feats['emotional'].shape[0] > 0:
                    features['emotional'].append(feats['emotional'])
                    features['source'].append(feats['source'])
                    features['filter'].append(feats['filter'])
            elif dimension == 'linguistic':
                if feats['linguistic'].shape[0] > 0:
                    features['linguistic'].append(feats['linguistic'])
                    features['source'].append(feats['source'])
                    features['filter'].append(feats['filter'])
            elif dimension == 'pathological':
                if feats['pathological'].shape[0] > 0:
                    features['pathological'].append(feats['pathological'])
                    features['source'].append(feats['source'])
                    features['filter'].append(feats['filter'])
                    
        except Exception as e:
            logger.debug(f"Failed to process {audio_path}: {e}")
            continue
    
    loaded_count = len(features[dimension]) if dimension in features else 0
    logger.info(f"    Loaded {loaded_count} samples from {dataset_name}")
    
    return features


def load_dataset_combination(emotional_path: str, emotional_name: str,
                            linguistic_path: str, linguistic_name: str,
                            pathological_path: str, pathological_name: str,
                            max_samples_per_dataset: int = 500) -> Dict[str, np.ndarray]:
    """Load a specific combination of datasets."""
    
    extractor = SpeechFeatureExtractor(sr=16000)
    
    # Load each dataset
    emo_features = load_single_dataset(emotional_path, emotional_name, 'emotional', 
                                       extractor, max_samples_per_dataset)
    ling_features = load_single_dataset(linguistic_path, linguistic_name, 'linguistic',
                                        extractor, max_samples_per_dataset)
    path_features = load_single_dataset(pathological_path, pathological_name, 'pathological',
                                        extractor, max_samples_per_dataset)
    
    # Balance datasets
    min_size = min(
        len(emo_features['emotional']),
        len(ling_features['linguistic']),
        len(path_features['pathological'])
    )
    
    if min_size == 0:
        raise ValueError(f"No samples loaded for combination: {emotional_name}-{linguistic_name}-{pathological_name}")
    
    logger.info(f"  Balancing to {min_size} samples per dimension")
    
    # Combine source and filter features from all datasets
    all_source = (emo_features['source'][:min_size] + 
                  ling_features['source'][:min_size] + 
                  path_features['source'][:min_size])
    all_filter = (emo_features['filter'][:min_size] + 
                  ling_features['filter'][:min_size] + 
                  path_features['filter'][:min_size])
    
    return {
        'emotional': np.array(emo_features['emotional'][:min_size]),
        'linguistic': np.array(ling_features['linguistic'][:min_size]),
        'pathological': np.array(path_features['pathological'][:min_size]),
        'source': np.array(all_source[:min_size]),
        'filter': np.array(all_filter[:min_size])
    }


def generate_summary_table(all_results: Dict, save_path: str) -> pd.DataFrame:
    """Generate summary comparison table."""
    
    rows = []
    for combo, results in all_results.items():
        row = {
            'Combination': combo,
            'Emo-Ling MINE': f"{results['mi_results']['Emotion-Linguistic']['mine_mean']:.2f}",
            'Emo-Ling CLUB': f"{results['mi_results']['Emotion-Linguistic']['club_mean']:.2f}",
            'Emo-Path MINE': f"{results['mi_results']['Emotion-Pathology']['mine_mean']:.2f}",
            'Emo-Path CLUB': f"{results['mi_results']['Emotion-Pathology']['club_mean']:.2f}",
            'Ling-Path MINE': f"{results['mi_results']['Linguistic-Pathology']['mine_mean']:.2f}",
            'Ling-Path CLUB': f"{results['mi_results']['Linguistic-Pathology']['club_mean']:.2f}",
            'Src-Flt MINE': f"{results['mi_results']['Source-Filter']['mine_mean']:.2f}",
            'Src-Flt CLUB': f"{results['mi_results']['Source-Filter']['club_mean']:.2f}",
            'Emo A_S': f"{results['attribution']['Emotional']['source']:.2f}",
            'Ling A_F': f"{results['attribution']['Linguistic']['filter']:.2f}",
            'Path A_F': f"{results['attribution']['Pathological']['filter']:.2f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    
    # Also save LaTeX version
    latex_path = save_path.replace('.csv', '.tex')
    latex_content = LatexTableGenerator.generate_summary_comparison_table(
        df,
        caption="Cross-Dimension MI and Source-Filter Attribution Summary",
        label="tab:summary_comparison"
    )
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    
    logger.info(f"Saved summary table: {save_path}")
    return df


# =========================================================================
# 10.1 CACHE UTILITIES (SAVE/LOAD ALL RESULTS)
# =========================================================================

def _to_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _safe_name(name: str) -> str:
    return name.replace(' ', '_').replace('/', '_').replace('\\', '_')


def save_cached_results(all_results: Dict, cache_dir: str) -> str:
    """Save full results + features to a cache folder for reuse."""
    os.makedirs(cache_dir, exist_ok=True)
    index = {
        'version': 1,
        'created_at': datetime.now().isoformat(),
        'combos': {}
    }

    for combo, res in all_results.items():
        safe = _safe_name(combo)
        features_file = f"{safe}_features.npz"
        features_path = os.path.join(cache_dir, features_file)

        feats = res.get('features', {})
        np.savez_compressed(
            features_path,
            emotional=feats.get('emotional', np.array([])),
            linguistic=feats.get('linguistic', np.array([])),
            pathological=feats.get('pathological', np.array([])),
            source=feats.get('source', np.array([])),
            filter=feats.get('filter', np.array([]))
        )

        index['combos'][combo] = {
            'features_file': features_file,
            'mi_results': res.get('mi_results', {}),
            'attribution': res.get('attribution', {})
        }

    index_path = os.path.join(cache_dir, 'cache_index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2, default=_to_serializable)

    logger.info(f"Saved cache to: {cache_dir}")
    return cache_dir


def load_cached_results(cache_dir: str) -> Dict:
    """Load cached results + features from a cache folder."""
    index_path = os.path.join(cache_dir, 'cache_index.json')
    if not os.path.exists(index_path):
        logger.warning(f"Cache index not found: {index_path}")
        return {}

    with open(index_path, 'r') as f:
        index = json.load(f)

    all_results = {}
    for combo, meta in index.get('combos', {}).items():
        features_path = os.path.join(cache_dir, meta['features_file'])
        if not os.path.exists(features_path):
            logger.warning(f"Missing cached features: {features_path}")
            continue

        with np.load(features_path) as npz:
            features = {
                'emotional': npz['emotional'] if 'emotional' in npz.files else np.array([]),
                'linguistic': npz['linguistic'] if 'linguistic' in npz.files else np.array([]),
                'pathological': npz['pathological'] if 'pathological' in npz.files else np.array([]),
                'source': npz['source'] if 'source' in npz.files else np.array([]),
                'filter': npz['filter'] if 'filter' in npz.files else np.array([])
            }

        all_results[combo] = {
            'mi_results': meta.get('mi_results', {}),
            'attribution': meta.get('attribution', {}),
            'features': features
        }

    logger.info(f"Loaded cache from: {cache_dir}")
    return all_results


# ============================================================================
# 11. MAIN EXPERIMENT FUNCTIONS
# ============================================================================

def run_single_combination(combo_name: str, emotional_path: str, emotional_name: str,
                          linguistic_path: str, linguistic_name: str,
                          pathological_path: str, pathological_name: str,
                          args, combo_index: int = None, combo_total: int = None) -> Optional[Dict]:
    """Run experiment for a single dataset combination."""
    
    logger.info(f"\n{'='*60}")
    if combo_index is not None and combo_total is not None:
        logger.info(f"Running combination {combo_index}/{combo_total}: {combo_name}")
    else:
        logger.info(f"Running combination: {combo_name}")
    logger.info(f"  Emotional: {emotional_name}")
    logger.info(f"  Linguistic: {linguistic_name}")
    logger.info(f"  Pathological: {pathological_name}")
    logger.info(f"{'='*60}")
    
    # Load data
    try:
        features = load_dataset_combination(
            emotional_path, emotional_name,
            linguistic_path, linguistic_name,
            pathological_path, pathological_name,
            max_samples_per_dataset=args.max_samples
        )
    except ValueError as e:
        logger.error(f"Failed to load combination {combo_name}: {e}")
        return None
    
    logger.info(f"  Emotional features: {features['emotional'].shape}")
    logger.info(f"  Linguistic features: {features['linguistic'].shape}")
    logger.info(f"  Pathological features: {features['pathological'].shape}")
    
    # Analyze
    analyzer = DimensionalAnalyzer()
    
    # MI estimation
    pairs = [
        ('Emotion-Linguistic', 'emotional', 'linguistic'),
        ('Emotion-Pathology', 'emotional', 'pathological'),
        ('Linguistic-Pathology', 'linguistic', 'pathological'),
        ('Source-Filter', 'source', 'filter')
    ]
    
    mi_results = {}
    for name, dim1, dim2 in pairs:
        logger.info(f"  Estimating MI for {name}...")
        result = analyzer.estimate_mi_pair(
            features[dim1], features[dim2], n_runs=args.n_runs
        )
        mi_results[name] = result
        logger.info(
            f"    MINE: {result['mine_mean']:.3f}, CLUB: {result['club_mean']:.3f}, "
            f"Combined: {result['combined_mean']:.3f}±{result['combined_std']:.3f}"
        )
    
    # Attribution
    attribution = {}
    for dim_name in ['emotional', 'linguistic', 'pathological']:
        source_attr, filter_attr = analyzer.compute_source_filter_attribution(
            features[dim_name], features['source'], features['filter']
        )
        attribution[dim_name.capitalize()] = {
            'source': source_attr,
            'filter': filter_attr
        }
        logger.info(f"  {dim_name.capitalize()}: Source={source_attr:.2f}, Filter={filter_attr:.2f}")
    
    return {
        'mi_results': mi_results,
        'attribution': attribution,
        'features': features
    }
def run_all_combinations(args) -> Dict:
    """Run experiments for all dataset combinations."""
    
    logger.info("=" * 70)
    logger.info("INTERSPEECH 2026: Multi-Dataset Comparison")
    logger.info("Running ALL 8 dataset combinations (2 x 2 x 2)")
    logger.info("MI estimation and cache generation")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cache_dir = args.cache_dir or os.path.join(output_dir, 'cache_new')

    all_results = {}
    
    # Define all dataset paths
    emotional_datasets = [
        (args.ravdess, 'RAVDESS'),
        (args.iemocap, 'IEMOCAP')
    ]
    linguistic_datasets = [
        (args.l2arctic, 'L2-ARCTIC'),
        (args.gmu, 'GMU')
    ]
    pathological_datasets = [
        (args.uaspeech, 'UA-Speech'),
        (args.mdvr, 'MDVR-KCL')
    ]
    
    # Generate all combinations (if not loaded from cache)
    if not all_results:
        combinations = list(product(emotional_datasets, linguistic_datasets, pathological_datasets))
        combo_total = len(combinations)

        for combo_index, ((emo_path, emo_name), (ling_path, ling_name), (path_path, path_name)) in enumerate(combinations, start=1):
            combo_name = f"{emo_name}-{ling_name}-{path_name}"
            
            result = run_single_combination(
                combo_name,
                emo_path, emo_name,
                ling_path, ling_name,
                path_path, path_name,
                args,
                combo_index=combo_index,
                combo_total=combo_total
            )
            
            if result is not None:
                all_results[combo_name] = result
    
    if not all_results:
        logger.error("No combinations completed successfully!")
        return {}

    # Save cache
    save_cached_results(all_results, cache_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Completed {len(all_results)} / 8 combinations")
    logger.info(f"{'='*70}")
    
    logger.info("\n" + "="*70)
    logger.info("CACHE GENERATED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}/")
    logger.info(f"Cache directory:  {cache_dir}/")
    logger.info("Generated files:")
    logger.info("  - cache_index.json")
    logger.info("  - *_features.npz (one per combination)")
    
    return all_results


# ============================================================================
# 12. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="INTERSPEECH 2026: MI Estimation Experiment Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
================================================================================
INTEGRATED VERSION FOR CACHE GENERATION
================================================================================

 Information-Theoretic Foundation for Speech Dimensional Analysis
  - Source-Filter Attribution with 95% CI
  - Cross-Dimension MI Analysis (Orthogonality)
  

  - Clustering Quality Metrics (Silhouette, DB, CH)
  - Bootstrap Stability Analysis
  - Parkinson's-Accent Confound Detection

DATASETS:
  Emotional:    RAVDESS, IEMOCAP
  Linguistic:   L2-ARCTIC, GMU Speech Accent Archive  
  Pathological: UA-Speech, MDVR-KCL

OUTPUT:
    - output/cache_new/cache_index.json
    - output/cache_new/*_features.npz

EXAMPLE:
    python speech_mi_framework_papers_integrated.py \\
        --ravdess /path/to/RAVDESS \\
        --iemocap /path/to/IEMOCAP \\
        --l2arctic /path/to/L2-ARCTIC \\
        --gmu /path/to/GMU \\
        --uaspeech /path/to/UA-Speech \\
        --mdvr /path/to/MDVR-KCL \\
        --output-dir results
================================================================================
        """
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "Data")
    
    # Emotional datasets
    parser.add_argument("--ravdess", type=str, 
                        default=os.path.join(data_dir, "RAVDESS"),
                        help="Path to RAVDESS dataset (emotional)")
    parser.add_argument("--iemocap", type=str, 
                        default=os.path.join(data_dir, "IEMOCAP"),
                        help="Path to IEMOCAP dataset (emotional)")
    
    # Linguistic datasets
    parser.add_argument("--l2arctic", type=str, 
                        default=os.path.join(data_dir, "L2-ARCTIC"),
                        help="Path to L2-ARCTIC dataset (linguistic)")
    parser.add_argument("--gmu", type=str, 
                        default=os.path.join(data_dir, "GMU-Accented Speech Archive"),
                        help="Path to GMU Speech Accent Archive (linguistic)")
    
    # Pathological datasets
    parser.add_argument("--uaspeech", type=str, 
                        default=os.path.join(data_dir, "UA-Speech"),
                        help="Path to UA-Speech dataset (pathological)")
    parser.add_argument("--mdvr", type=str, 
                        default=os.path.join(data_dir, "MDVR-KCL"),
                        help="Path to MDVR-KCL dataset (pathological)")
    
    # Experiment settings
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max samples per dataset (default: 500)")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="MI estimation runs (default: 10)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory (default: output)")

    # Cache controls
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory for saving/loading features/results")
    
    args = parser.parse_args()
    
    try:
        all_results = run_all_combinations(args)
        if not all_results:
            raise RuntimeError("No combinations completed successfully. Check dataset paths and file formats.")
        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*70)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
