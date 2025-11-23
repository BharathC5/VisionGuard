#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid Neuro + GA + Fuzzy Pipeline
==================================

Data:
    - train_imgs/       + train_labels.csv
    - ttrain_imgs/      + ttrain_labels.csv
    - test_imgs/        + test_labels.csv

Goal:
    neuro (CNN) + GA (hyperparam + fuzzy params) -> optimized op
    then fuzzy system refines final decision (neuro + fuzzy -> final op)

Assumptions about CSVs:
    Each CSV has columns:
        image_id   (without extension, e.g., "0001")
        label      (integer DR grade: 0..4 or any multi-class)

Adjust CSV column names / paths as needed for your project.

This script is LARGE and modular:
    - Config & utils
    - Data loading + augmentation
    - CNN model (EfficientNet-B0)
    - GA hyperparameter search
    - Fuzzy system (triangular MFs + rules)
    - Final evaluation on test set

Designed for:
    - GPU: NVIDIA RTX 4070 Ti
    - Mixed precision enabled (torch.cuda.amp)
"""

import os
import math
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class Config:
    # Paths
    train_img_dir: str = "train_imgs"
    ttrain_img_dir: str = "ttrain_imgs"
    test_img_dir: str = "test_imgs"

    train_csv: str = "train_labels.csv"
    ttrain_csv: str = "ttrain_labels.csv"
    test_csv: str = "test_labels.csv"

    # Image & model
    img_size: int = 300          # 300x300 images
    num_classes: int = 5         # e.g., DR grade 0..4
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    feature_dim: int = 1024      # after projection
    dropout_default: float = 0.3

    # Training / eval
    base_lr: float = 1e-3
    base_weight_decay: float = 1e-4
    base_batch_size: int = 32

    epochs_ga_eval: int = 4      # short training for GA
    epochs_full_train: int = 20  # full training

    # GA settings
    ga_population: int = 14
    ga_generations: int = 8
    ga_mutation_rate: float = 0.25
    ga_crossover_rate: float = 0.8
    ga_elitism_k: int = 2

    # Fuzzy system
    num_fuzzy_features: int = 4   # severity, entropy, vessel, texture
    fuzzy_sets_per_feature: int = 3  # low, mid, high

    # Device & AMP
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True

    # Random seeds
    seed: int = 42

    # Splits
    val_ratio: float = 0.2

    # Checkpoints / logs
    ckpt_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_neuro_model.pt"
    best_ga_state_path: str = "checkpoints/best_ga_state.json"

    # Logging
    verbose: bool = True


cfg = Config()


# ============================================================
# 2. UTILS
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg: str):
    if cfg.verbose:
        print(msg)


def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv_pairs(csv_path: str, img_dir: str) -> List[Tuple[str, int]]:
    """
    Load (image_path, label) pairs from CSV.

    Expects columns:
        image_id    (without ext)
        label       (int)
    """
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain columns: ['image_id', 'label']. Found: {df.columns}"
        )
    pairs = []
    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        label = int(row["label"])

        # Assume .png, .jpg, .jpeg; adjust if needed
        possible_exts = [".png", ".jpg", ".jpeg"]
        found_path = None
        for ext in possible_exts:
            p = os.path.join(img_dir, image_id + ext)
            if os.path.isfile(p):
                found_path = p
                break
        if found_path is None:
            # fallback: maybe image_id already has extension
            p = os.path.join(img_dir, image_id)
            if not os.path.isfile(p):
                log(f"[WARN] Could not find image for id {image_id} in {img_dir}")
                continue
            found_path = p

        pairs.append((found_path, label))
    return pairs


# ============================================================
# 3. DATASET & TRANSFORMS
# ============================================================

class DRDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(aug_intensity: float = 0.5):
    """
    aug_intensity in [0, 1].
    Higher → stronger augmentation.
    """
    sz = cfg.img_size
    # Map intensity to params
    max_deg = 10 + 20 * aug_intensity
    color_factor = 0.1 + 0.3 * aug_intensity

    train_tfms = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(p=0.5 * aug_intensity),
        transforms.RandomVerticalFlip(p=0.2 * aug_intensity),
        transforms.RandomRotation(degrees=max_deg),
        transforms.ColorJitter(
            brightness=color_factor,
            contrast=color_factor,
            saturation=color_factor
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.25, 0.25, 0.25]),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.25, 0.25, 0.25]),
    ])

    return train_tfms, test_tfms


# ============================================================
# 4. MODEL: EFFICIENTNET-B0 BACKBONE
# ============================================================

class NeuroBackbone(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 1024,
                 dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        # EfficientNet-B0 backbone
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        # Replace classifier
        self.backbone.classifier = nn.Identity()

        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)               # [B, in_features]
        feats = self.feature_proj(x)       # [B, feature_dim]
        logits = self.classifier(feats)    # [B, num_classes]
        return logits, feats


# ============================================================
# 5. METRICS
# ============================================================

def compute_kappa(y_true, y_pred):
    if len(set(y_true)) <= 1:
        return 0.0
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def eval_model(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits, _ = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
    acc = accuracy_score(all_true, all_pred)
    kappa = compute_kappa(all_true, all_pred)
    return acc, kappa


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [B, C], softmax outputs.
    Returns entropy per sample (B,).
    """
    eps = 1e-8
    logp = torch.log(probs + eps)
    ent = -torch.sum(probs * logp, dim=1)
    return ent


# ============================================================
# 6. SIMPLE PLACEHOLDER FEATURE FUNCTIONS
# ============================================================

def compute_vessel_density(images: torch.Tensor) -> torch.Tensor:
    """
    Placeholder: use mean intensity as a proxy.
    images: [B, C, H, W]
    """
    # convert to [0,1] approx; already normalized but it's ok as relative measure
    v = images.mean(dim=[1, 2, 3])  # [B]
    # map to [0,1] via sigmoid
    return torch.sigmoid(3 * v)


def compute_texture_score(features: torch.Tensor) -> torch.Tensor:
    """
    Placeholder: use feature L2 norm as texture complexity proxy.
    features: [B, F]
    """
    norm = torch.norm(features, dim=1)  # [B]
    # Normalize with softsign
    return norm / (1.0 + torch.abs(norm))


# ============================================================
# 7. CHROMOSOME & GA
# ============================================================

class Chromosome:
    def __init__(self):
        # hyperparams
        self.lr = None
        self.weight_decay = None
        self.batch_size = None
        self.dropout = None
        self.aug_intensity = None

        # fuzzy parameters: flattened [a,b,c,a,b,c,...] per feature+set
        self.fuzzy_params: List[float] = []

        self.fitness: float = float("-inf")

    def to_dict(self):
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "aug_intensity": self.aug_intensity,
            "fuzzy_params": self.fuzzy_params,
            "fitness": self.fitness,
        }

    @staticmethod
    def from_dict(d: Dict):
        c = Chromosome()
        c.lr = d["lr"]
        c.weight_decay = d["weight_decay"]
        c.batch_size = d["batch_size"]
        c.dropout = d["dropout"]
        c.aug_intensity = d["aug_intensity"]
        c.fuzzy_params = d["fuzzy_params"]
        c.fitness = d["fitness"]
        return c


def sample_log_uniform(lo: float, hi: float) -> float:
    return float(math.exp(random.uniform(math.log(lo), math.log(hi))))


def init_random_chromosome() -> Chromosome:
    c = Chromosome()
    c.lr = sample_log_uniform(1e-5, 1e-2)
    c.weight_decay = sample_log_uniform(1e-6, 1e-3)
    c.batch_size = random.choice([16, 24, 32, 40])
    c.dropout = random.uniform(0.1, 0.6)
    c.aug_intensity = random.uniform(0.1, 1.0)

    total_triangles = cfg.num_fuzzy_features * cfg.fuzzy_sets_per_feature
    for _ in range(total_triangles):
        a = random.uniform(0.0, 0.33)
        b = random.uniform(0.33, 0.66)
        c_param = random.uniform(0.66, 1.0)
        c.fuzzy_params.extend([a, b, c_param])

    return c


def init_population(n: int) -> List[Chromosome]:
    return [init_random_chromosome() for _ in range(n)]


# ============================================================
# 8. GA FITNESS EVALUATION (FAST TRAIN)
# ============================================================

def evaluate_chromosome(chrom: Chromosome,
                        train_samples: List[Tuple[str, int]],
                        val_samples: List[Tuple[str, int]]) -> float:
    """
    Short training loop to approximate fitness of a chromosome.
    Fitness = validation kappa + 0.1 * fuzzy_consistency
    """
    # Build transforms
    train_tfms, val_tfms = build_transforms(aug_intensity=chrom.aug_intensity)

    train_ds = DRDataset(train_samples, transform=train_tfms)
    val_ds = DRDataset(val_samples, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=chrom.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=chrom.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = NeuroBackbone(
        num_classes=cfg.num_classes,
        feature_dim=cfg.feature_dim,
        dropout=chrom.dropout,
        pretrained=cfg.pretrained,
    )
    model.to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=chrom.lr,
        weight_decay=chrom.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_kappa = float("-inf")

    for epoch in range(1, cfg.epochs_ga_eval + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        acc, kappa = eval_model(model, val_loader, cfg.device)
        best_kappa = max(best_kappa, kappa)

    # Fuzzy consistency
    fuzzy_consistency = compute_fuzzy_consistency(
        model, val_loader, chrom.fuzzy_params
    )

    chrom.fitness = best_kappa + 0.1 * fuzzy_consistency
    log(f"[GA] fitness = {chrom.fitness:.4f}, kappa={best_kappa:.4f}, fuzzy={fuzzy_consistency:.4f}")
    return chrom.fitness


# ============================================================
# 9. FUZZY MEMBERSHIP FUNCTIONS
# ============================================================

def build_fuzzy_membership_functions(fuzzy_params: List[float]) -> Dict[int, List[Tuple[float, float, float]]]:
    """
    fuzzy_params: flattened [a,b,c,a,b,c,...] for each feature × set.
    Returns dict: feature_idx -> list of (a,b,c) for sets [LOW,MID,HIGH].
    """
    fuzzy_sets = {}
    idx = 0
    for feat_idx in range(cfg.num_fuzzy_features):
        fuzzy_sets[feat_idx] = []
        for _ in range(cfg.fuzzy_sets_per_feature):
            a = fuzzy_params[idx]; b = fuzzy_params[idx+1]; c_val = fuzzy_params[idx+2]
            idx += 3
            trip = sorted([a, b, c_val])
            fuzzy_sets[feat_idx].append(tuple(trip))
    return fuzzy_sets


def tri_membership(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a + 1e-8)
    elif b < x < c:
        return (c - x) / (c - b + 1e-8)
    return 0.0


# risk output sets: indices 0=LOW,1=MED,2=HIGH,3=VERY_HIGH
def risk_output_fuzzy_set(idx: int) -> Tuple[float, float, float]:
    if idx == 0:   # LOW
        return (0.0, 0.0, 0.3)
    elif idx == 1: # MED
        return (0.2, 0.4, 0.6)
    elif idx == 2: # HIGH
        return (0.5, 0.7, 0.9)
    else:          # VERY_HIGH
        return (0.8, 1.0, 1.0)


# ============================================================
# 10. FUZZY RULES
# ============================================================

class FuzzyRule:
    def __init__(self, antecedent, consequent: int):
        """
        antecedent: list of (feature_idx, fuzzy_label_idx), fuzzy_label_idx in {0,1,2} (LOW/MID/HIGH)
        consequent: index for risk fuzzy set (0..3)
        """
        self.antecedent = antecedent
        self.consequent = consequent


def define_fuzzy_rules() -> List[FuzzyRule]:
    """
    Example: 4 fuzzy features:
        f0 = severity
        f1 = entropy
        f2 = vessel_density
        f3 = texture_score
    fuzzy_label_idx: 0=LOW,1=MID,2=HIGH
    """
    rules = []

    # Rule1: IF severity HIGH AND entropy LOW THEN risk VERY_HIGH
    rules.append(FuzzyRule(
        antecedent=[(0, 2), (1, 0)],
        consequent=3
    ))

    # Rule2: IF severity MID AND entropy MID AND vessel LOW THEN risk MED
    rules.append(FuzzyRule(
        antecedent=[(0, 1), (1, 1), (2, 0)],
        consequent=1
    ))

    # Rule3: IF severity LOW AND entropy HIGH THEN risk MED
    rules.append(FuzzyRule(
        antecedent=[(0, 0), (1, 2)],
        consequent=1
    ))

    # Rule4: IF severity LOW AND entropy LOW THEN risk LOW
    rules.append(FuzzyRule(
        antecedent=[(0, 0), (1, 0)],
        consequent=0
    ))

    # Rule5: IF severity HIGH AND vessel HIGH THEN risk HIGH
    rules.append(FuzzyRule(
        antecedent=[(0, 2), (2, 2)],
        consequent=2
    ))

    return rules


# ============================================================
# 11. FUZZY INFERENCE (MAMDANI)
# ============================================================

def fuzzy_inference_single_sample(
    fv: np.ndarray,
    fuzzy_sets: Dict[int, List[Tuple[float, float, float]]],
    rules: List[FuzzyRule]
) -> float:
    """
    fv: array of length num_fuzzy_features, normalized in [0,1].
    Returns crisp risk value in [0,1].
    """
    # Step1: membership degrees
    memberships = {feat_idx: [] for feat_idx in range(cfg.num_fuzzy_features)}
    for feat_idx in range(cfg.num_fuzzy_features):
        x = float(fv[feat_idx])
        for (a, b, c) in fuzzy_sets[feat_idx]:
            memberships[feat_idx].append(tri_membership(x, a, b, c))

    # Step2: rule firing strengths (min of antecedent)
    rule_strengths = []
    rule_outputs = []
    for rule in rules:
        mus = []
        for feat_idx, set_idx in rule.antecedent:
            mus.append(memberships[feat_idx][set_idx])
        strength = min(mus) if mus else 0.0
        rule_strengths.append(strength)
        rule_outputs.append(rule.consequent)

    # Step3: aggregation of consequents
    R = 101
    z_values = np.linspace(0.0, 1.0, R)
    aggregated_mu = np.zeros_like(z_values)

    for r_idx, strength in enumerate(rule_strengths):
        cons_idx = rule_outputs[r_idx]
        a, b, c = risk_output_fuzzy_set(cons_idx)
        for i, z in enumerate(z_values):
            mu_out = tri_membership(z, a, b, c)
            aggregated_mu[i] = max(aggregated_mu[i], min(strength, mu_out))

    # Step4: centroid defuzzification
    if np.sum(aggregated_mu) < 1e-8:
        return 0.0
    crisp = np.sum(z_values * aggregated_mu) / (np.sum(aggregated_mu) + 1e-8)
    return float(crisp)


def normalize_fuzzy_features(fv: np.ndarray) -> np.ndarray:
    """
    Normalize fuzzy feature vector to [0,1] per dimension.
    Simple min-max with preset ranges; adjust for real stats.
    """
    # naive: clip 0..1
    return np.clip(fv, 0.0, 1.0)


# ============================================================
# 12. FUZZY CONSISTENCY SCORE FOR GA FITNESS
# ============================================================

def compute_fuzzy_consistency(model: nn.Module,
                              val_loader: DataLoader,
                              fuzzy_params: List[float]) -> float:
    model.eval()
    fuzzy_sets = build_fuzzy_membership_functions(fuzzy_params)
    rules = define_fuzzy_rules()

    labels_all: List[int] = []
    risks_all: List[float] = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)

            logits, feats = model(imgs)
            probs = F.softmax(logits, dim=1)
            severity = probs[:, -1]         # prob of last class as severity
            ent = entropy_from_probs(probs)
            vessel = compute_vessel_density(imgs)
            texture = compute_texture_score(feats)

            for i in range(imgs.size(0)):
                fv = np.array([
                    float(severity[i].cpu()),
                    float(ent[i].cpu() / math.log(cfg.num_classes + 1e-8)),  # normalize entropy
                    float(vessel[i].cpu()),
                    float(texture[i].cpu()),
                ], dtype=np.float32)
                fv_norm = normalize_fuzzy_features(fv)
                crisp_risk = fuzzy_inference_single_sample(fv_norm, fuzzy_sets, rules)
                labels_all.append(int(labels[i].cpu()))
                risks_all.append(crisp_risk)

    if len(set(labels_all)) <= 1:
        return 0.0

    corr, _ = spearmanr(labels_all, risks_all)
    if corr is None or math.isnan(corr):
        return 0.0
    consistency = (corr + 1.0) / 2.0  # map [-1,1] -> [0,1]
    return float(consistency)


# ============================================================
# 13. GA OPERATORS
# ============================================================

def tournament_selection(pop: List[Chromosome], k: int = 3) -> Chromosome:
    cand = random.choice(pop)
    for _ in range(k - 1):
        challenger = random.choice(pop)
        if challenger.fitness > cand.fitness:
            cand = challenger
    # clone
    new_c = Chromosome()
    new_c.lr = cand.lr
    new_c.weight_decay = cand.weight_decay
    new_c.batch_size = cand.batch_size
    new_c.dropout = cand.dropout
    new_c.aug_intensity = cand.aug_intensity
    new_c.fuzzy_params = cand.fuzzy_params.copy()
    new_c.fitness = cand.fitness
    return new_c


def mutate(chrom: Chromosome) -> Chromosome:
    if random.random() < cfg.ga_mutation_rate:
        chrom.lr *= math.exp(random.gauss(0, 0.3))
        chrom.lr = float(np.clip(chrom.lr, 1e-6, 1e-2))
    if random.random() < cfg.ga_mutation_rate:
        chrom.weight_decay *= math.exp(random.gauss(0, 0.3))
        chrom.weight_decay = float(np.clip(chrom.weight_decay, 1e-7, 1e-3))
    if random.random() < cfg.ga_mutation_rate:
        chrom.batch_size = random.choice([16, 24, 32, 40])
    if random.random() < cfg.ga_mutation_rate:
        chrom.dropout += random.gauss(0, 0.05)
        chrom.dropout = float(np.clip(chrom.dropout, 0.05, 0.7))
    if random.random() < cfg.ga_mutation_rate:
        chrom.aug_intensity += random.gauss(0, 0.1)
        chrom.aug_intensity = float(np.clip(chrom.aug_intensity, 0.1, 1.0))

    for i in range(len(chrom.fuzzy_params)):
        if random.random() < (cfg.ga_mutation_rate / 2.0):
            chrom.fuzzy_params[i] += random.gauss(0, 0.03)
            chrom.fuzzy_params[i] = float(np.clip(chrom.fuzzy_params[i], 0.0, 1.0))

    chrom.fitness = float("-inf")
    return chrom


def crossover(p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    c1 = Chromosome()
    c2 = Chromosome()

    # Flatten gene vector except fuzzy_params
    if random.random() < cfg.ga_crossover_rate:
        # hyperparams single point
        # (lr, wd, bs, dp, aug)
        genes1 = [p1.lr, p1.weight_decay, float(p1.batch_size),
                  p1.dropout, p1.aug_intensity]
        genes2 = [p2.lr, p2.weight_decay, float(p2.batch_size),
                  p2.dropout, p2.aug_intensity]
        point = random.randint(1, len(genes1) - 1)
        new_g1 = genes1[:point] + genes2[point:]
        new_g2 = genes2[:point] + genes1[point:]
    else:
        new_g1 = [p1.lr, p1.weight_decay, float(p1.batch_size),
                  p1.dropout, p1.aug_intensity]
        new_g2 = [p2.lr, p2.weight_decay, float(p2.batch_size),
                  p2.dropout, p2.aug_intensity]

    c1.lr, c1.weight_decay, bs1, c1.dropout, c1.aug_intensity = new_g1
    c2.lr, c2.weight_decay, bs2, c2.dropout, c2.aug_intensity = new_g2
    c1.batch_size = int(bs1)
    c2.batch_size = int(bs2)

    # fuzzy params uniform crossover
    c1.fuzzy_params = []
    c2.fuzzy_params = []
    for fp1, fp2 in zip(p1.fuzzy_params, p2.fuzzy_params):
        if random.random() < 0.5:
            c1.fuzzy_params.append(fp1)
            c2.fuzzy_params.append(fp2)
        else:
            c1.fuzzy_params.append(fp2)
            c2.fuzzy_params.append(fp1)

    c1.fitness = float("-inf")
    c2.fitness = float("-inf")
    return c1, c2


# ============================================================
# 14. GA MAIN LOOP
# ============================================================

def save_ga_state(chrom: Chromosome, path: str):
    ensure_dirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(chrom.to_dict(), f, indent=2)


def load_ga_state(path: str) -> Optional[Chromosome]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        d = json.load(f)
    return Chromosome.from_dict(d)


def run_ga(train_samples: List[Tuple[str, int]],
           val_samples: List[Tuple[str, int]]) -> Chromosome:
    log("=== GA START ===")
    population = init_population(cfg.ga_population)
    best_overall: Optional[Chromosome] = None

    for gen in range(1, cfg.ga_generations + 1):
        log(f"[GA] Generation {gen}/{cfg.ga_generations}")

        # Evaluate
        for chrom in population:
            if chrom.fitness == float("-inf"):
                evaluate_chromosome(chrom, train_samples, val_samples)

        population.sort(key=lambda c: c.fitness, reverse=True)

        if best_overall is None or population[0].fitness > best_overall.fitness:
            best_overall = population[0]
            log(f"[GA] New best fitness: {best_overall.fitness:.4f}")
            save_ga_state(best_overall, cfg.best_ga_state_path)

        # Elitism
        new_pop: List[Chromosome] = []
        for i in range(cfg.ga_elitism_k):
            elite = population[i]
            c = Chromosome()
            c.lr = elite.lr
            c.weight_decay = elite.weight_decay
            c.batch_size = elite.batch_size
            c.dropout = elite.dropout
            c.aug_intensity = elite.aug_intensity
            c.fuzzy_params = elite.fuzzy_params.copy()
            c.fitness = elite.fitness
            new_pop.append(c)

        # Fill rest
        while len(new_pop) < cfg.ga_population:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < cfg.ga_population:
                new_pop.append(c2)

        population = new_pop

    log("=== GA END ===")
    return best_overall


# ============================================================
# 15. FULL TRAIN WITH BEST GA CHROMOSOME
# ============================================================

def train_full_model(chrom: Chromosome,
                     all_train_samples: List[Tuple[str, int]]) -> nn.Module:
    log("=== FULL TRAINING WITH GA-OPTIMIZED HYPERPARAMS ===")
    train_samples, val_samples = train_test_split(
        all_train_samples,
        test_size=cfg.val_ratio,
        stratify=[l for _, l in all_train_samples],
        random_state=cfg.seed
    )

    train_tfms, val_tfms = build_transforms(aug_intensity=chrom.aug_intensity)

    train_ds = DRDataset(train_samples, transform=train_tfms)
    val_ds = DRDataset(val_samples, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=chrom.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=chrom.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    model = NeuroBackbone(
        num_classes=cfg.num_classes,
        feature_dim=cfg.feature_dim,
        dropout=chrom.dropout,
        pretrained=cfg.pretrained,
    )
    model.to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=chrom.lr,
        weight_decay=chrom.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_kappa = float("-inf")
    ensure_dirs(cfg.ckpt_dir)

    for epoch in range(1, cfg.epochs_full_train + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc, val_kappa = eval_model(model, val_loader, cfg.device)

        log(f"[EPOCH {epoch}] train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, val_kappa={val_kappa:.4f}")

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), cfg.best_model_path)
            log(f"[EPOCH {epoch}] New best model saved (kappa={best_kappa:.4f})")

    # load best
    best_model = NeuroBackbone(
        num_classes=cfg.num_classes,
        feature_dim=cfg.feature_dim,
        dropout=chrom.dropout,
        pretrained=cfg.pretrained,
    )
    best_model.load_state_dict(torch.load(cfg.best_model_path, map_location=cfg.device))
    best_model.to(cfg.device)
    return best_model


# ============================================================
# 16. FINAL TEST EVAL WITH NEURO + FUZZY COMBO
# ============================================================

def evaluate_on_test(model: nn.Module,
                     chrom: Chromosome,
                     test_samples: List[Tuple[str, int]]):
    _, test_tfms = build_transforms(aug_intensity=chrom.aug_intensity)

    test_ds = DRDataset(test_samples, transform=test_tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=chrom.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    fuzzy_sets = build_fuzzy_membership_functions(chrom.fuzzy_params)
    rules = define_fuzzy_rules()

    all_true = []
    all_base = []
    all_final = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)

            logits, feats = model(imgs)
            probs = F.softmax(logits, dim=1)
            base_preds = torch.argmax(probs, dim=1)

            severity = probs[:, -1]
            ent = entropy_from_probs(probs)
            vessel = compute_vessel_density(imgs)
            texture = compute_texture_score(feats)

            for i in range(imgs.size(0)):
                fv = np.array([
                    float(severity[i].cpu()),
                    float(ent[i].cpu() / math.log(cfg.num_classes + 1e-8)),
                    float(vessel[i].cpu()),
                    float(texture[i].cpu()),
                ], dtype=np.float32)
                fv_norm = normalize_fuzzy_features(fv)
                risk = fuzzy_inference_single_sample(fv_norm, fuzzy_sets, rules)

                base_c = int(base_preds[i].cpu())
                final_c = base_c

                # Example combination logic: adjust for high/low risk
                if risk > 0.8 and base_c < cfg.num_classes - 1:
                    final_c = base_c + 1
                if risk < 0.2 and base_c > 0:
                    final_c = base_c - 1

                all_true.append(int(labels[i].cpu()))
                all_base.append(base_c)
                all_final.append(final_c)

    base_acc = accuracy_score(all_true, all_base)
    final_acc = accuracy_score(all_true, all_final)
    base_kappa = compute_kappa(all_true, all_base)
    final_kappa = compute_kappa(all_true, all_final)

    log("=== TEST RESULTS ===")
    log(f"Base  model: acc={base_acc:.4f}, kappa={base_kappa:.4f}")
    log(f"Final model: acc={final_acc:.4f}, kappa={final_kappa:.4f}")
    log("Classification report (final):")
    log(classification_report(all_true, all_final, digits=4))


# ============================================================
# 17. MAIN
# ============================================================

def main():
    set_seed(cfg.seed)
    ensure_dirs(cfg.ckpt_dir)
    log(f"Using device: {cfg.device}")

    # 1. Load data from all CSVs
    train_pairs = load_csv_pairs(cfg.train_csv, cfg.train_img_dir)
    ttrain_pairs = load_csv_pairs(cfg.ttrain_csv, cfg.ttrain_img_dir)
    test_pairs = load_csv_pairs(cfg.test_csv, cfg.test_img_dir)

    # merge train + ttrain
    all_train_pairs = train_pairs + ttrain_pairs
    all_labels = [l for _, l in all_train_pairs]

    # initial split for GA (train/val)
    train_ids, val_ids = train_test_split(
        list(range(len(all_train_pairs))),
        test_size=cfg.val_ratio,
        stratify=all_labels,
        random_state=cfg.seed,
    )
    ga_train_samples = [all_train_pairs[i] for i in train_ids]
    ga_val_samples = [all_train_pairs[i] for i in val_ids]

    # 2. Run GA or load existing
    best_chrom = load_ga_state(cfg.best_ga_state_path)
    if best_chrom is None:
        best_chrom = run_ga(ga_train_samples, ga_val_samples)
        log(f"Best GA fitness: {best_chrom.fitness:.4f}")
    else:
        log(f"Loaded GA state with fitness: {best_chrom.fitness:.4f}")

    # 3. Full training
    best_model = train_full_model(best_chrom, all_train_pairs)

    # 4. Test evaluation (neuro + fuzzy)
    evaluate_on_test(best_model, best_chrom, test_pairs)


if __name__ == "__main__":
    main()
