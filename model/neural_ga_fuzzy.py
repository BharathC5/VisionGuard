#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hybrid_neuro_ga_fuzzy_export.py
Exported runnable script: Hybrid Neuro + GA + Fuzzy pipeline
Includes:
 - status prints and CUDA checks
 - multi-checkpoint resume (checkpoint_epoch_XX.pt per epoch)
 - Windows-safe dataloader (num_workers=0)
 - uses torch.amp new API (GradScaler + autocast)
 - EfficientNet-B0 backbone
 - GA (fast eval) and full training loops
 - fuzzy inference (Mamdani)
Usage:
    python hybrid_neuro_ga_fuzzy_export.py --work-dir "C:/Users/Aditya/Downloads/APTOS-2019 Dataset"
"""

import os
import sys
import math
import time
import json
import random
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, models

from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Config dataclass
# ---------------------------
@dataclass
class Config:
    work_dir: str = "."
    train_img_dir: str = "train_images"
    val_img_dir: str = "val_images"
    test_img_dir: str = "test_images"

    train_csv: str = "train_1.csv"
    val_csv: str = "valid.csv"
    test_csv: str = "test.csv"

    img_size: int = 300
    num_classes: int = 5
    feature_dim: int = 1024

    # GA
    ga_population: int = 8
    ga_generations: int = 5
    ga_mutation_rate: float = 0.25
    ga_crossover_rate: float = 0.8
    ga_elitism_k: int = 2
    epochs_ga_eval: int = 2

    # Training
    epochs_full_train: int = 12
    base_batch_size: int = 24

    # fuzzy
    num_fuzzy_features: int = 4
    fuzzy_sets_per_feature: int = 3

    # device & amp
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True

    # resume & checkpoints
    ckpt_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pt"

    # misc
    seed: int = 42
    val_ratio: float = 0.2
    verbose: bool = True

cfg = Config()

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # keep cudnn fast
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def log(msg: str, force: bool = False):
    if cfg.verbose or force:
        print(msg)

def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def print_cuda_banner():
    print("\n" + "="*50)
    print(f" CUDA available: {torch.cuda.is_available()} ")
    if torch.cuda.is_available():
        try:
            print(f" GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    else:
        print(" RUNNING ON CPU — training will be much slower.")
    print("="*50 + "\n")

# ---------------------------
# CSV loader
# ---------------------------
def load_csv_pairs(csv_path: str, img_dir: str) -> List[Tuple[str, int]]:
    if not os.path.isfile(csv_path):
        log(f"[WARN] CSV not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    # infer columns
    img_col = None
    for c in df.columns:
        if c.lower() in ("image", "image_id", "id_code", "id"):
            img_col = c; break
    if img_col is None: img_col = df.columns[0]
    label_col = None
    for c in df.columns:
        if c.lower() in ("label", "level", "diagnosis", "dr_grade"):
            label_col = c; break
    if label_col is None:
        if len(df.columns) > 1:
            label_col = df.columns[1]
        else:
            raise ValueError(f"Cannot infer label column in {csv_path}")

    pairs = []
    for _, row in df.iterrows():
        image_id = str(row[img_col])
        try:
            label = int(row[label_col])
        except Exception:
            label = int(float(row[label_col]))
        # try common extensions
        found = None
        for ext in (".png", ".jpg", ".jpeg", ""):
            p = os.path.join(img_dir, image_id + ext)
            if os.path.isfile(p):
                found = p; break
            # maybe image_id contains extension
            p2 = os.path.join(img_dir, os.path.basename(image_id))
            if os.path.isfile(p2):
                found = p2; break
        if found is None:
            log(f"[WARN] missing image: {image_id} in {img_dir}")
            continue
        pairs.append((found, label))
    return pairs

# ---------------------------
# Dataset & transforms
# ---------------------------
class DRDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            log(f"[ERROR] Failed to load {p}: {e}")
            img = Image.new('RGB', (cfg.img_size, cfg.img_size))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)

def build_transforms(aug_intensity: float = 0.6):
    sz = cfg.img_size
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(sz, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_tfms, test_tfms

# ---------------------------
# Model
# ---------------------------
class NeuroBackbone(nn.Module):
    def __init__(self, num_classes: int = None, feature_dim: int = None, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        if num_classes is None: num_classes = cfg.num_classes
        if feature_dim is None: feature_dim = cfg.feature_dim
        # use torchvision weights API if available
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            base = models.efficientnet_b0(weights=weights)
        except Exception:
            base = models.efficientnet_b0(weights=None)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        feats = self.feature_proj(x)
        logits = self.classifier(feats)
        return logits, feats

# ---------------------------
# Metrics and helpers
# ---------------------------
def compute_kappa(y_true, y_pred):
    if len(set(y_true)) <= 1: return 0.0
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def eval_model(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device); labels = labels.to(device)
            logits, _ = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    acc = accuracy_score(all_true, all_pred)
    kappa = compute_kappa(all_true, all_pred)
    return acc, kappa

def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    logp = torch.log(probs + eps)
    ent = -torch.sum(probs * logp, dim=1)
    return ent

def compute_vessel_density(images: torch.Tensor) -> torch.Tensor:
    v = images.mean(dim=[1,2,3])
    return torch.sigmoid(3 * v)

def compute_texture_score(features: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(features, dim=1)
    return norm / (1.0 + torch.abs(norm))

# ---------------------------
# Chromosome & GA (simplified but functional)
# ---------------------------
class Chromosome:
    def __init__(self):
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = cfg.base_batch_size
        self.dropout = 0.3
        self.aug_intensity = 0.6
        self.fuzzy_params = []
        total_triangles = cfg.num_fuzzy_features * cfg.fuzzy_sets_per_feature
        for _ in range(total_triangles):
            a = random.uniform(0.0, 0.33)
            b = random.uniform(0.33, 0.66)
            cparam = random.uniform(0.66, 1.0)
            self.fuzzy_params.extend([a,b,cparam])
        self.fitness = float('-inf')

    def to_dict(self):
        return {
            'lr': self.lr, 'weight_decay': self.weight_decay, 'batch_size': self.batch_size,
            'dropout': self.dropout, 'aug_intensity': self.aug_intensity,
            'fuzzy_params': self.fuzzy_params, 'fitness': self.fitness
        }

    @staticmethod
    def from_dict(d):
        c = Chromosome()
        c.lr = d.get('lr', c.lr)
        c.weight_decay = d.get('weight_decay', c.weight_decay)
        c.batch_size = d.get('batch_size', c.batch_size)
        c.dropout = d.get('dropout', c.dropout)
        c.aug_intensity = d.get('aug_intensity', c.aug_intensity)
        c.fuzzy_params = d.get('fuzzy_params', c.fuzzy_params)
        c.fitness = d.get('fitness', c.fitness)
        return c

def sample_log_uniform(lo, hi):
    return float(math.exp(random.uniform(math.log(lo), math.log(hi))))

def init_random_chromosome():
    c = Chromosome()
    c.lr = sample_log_uniform(1e-5, 1e-3)
    c.weight_decay = sample_log_uniform(1e-6, 1e-3)
    c.batch_size = random.choice([16, 24, 32, 40])
    c.dropout = random.uniform(0.1, 0.6)
    c.aug_intensity = random.uniform(0.1, 1.0)
    return c

def init_population(n):
    return [init_random_chromosome() for _ in range(n)]

# fuzzy functions (as before)
def build_fuzzy_membership_functions(fuzzy_params):
    fuzzy_sets = {}
    idx = 0
    for feat_idx in range(cfg.num_fuzzy_features):
        fuzzy_sets[feat_idx] = []
        for _ in range(cfg.fuzzy_sets_per_feature):
            a = fuzzy_params[idx]; b = fuzzy_params[idx+1]; cval = fuzzy_params[idx+2]
            idx += 3
            fuzzy_sets[feat_idx].append(tuple(sorted([a,b,cval])))
    return fuzzy_sets

def tri_membership(x,a,b,c):
    if x <= a or x >= c: return 0.0
    if abs(x-b) < 1e-9: return 1.0
    if a < x < b: return (x-a)/(b-a+1e-8)
    if b < x < c: return (c-x)/(c-b+1e-8)
    return 0.0

def risk_output_fuzzy_set(idx):
    if idx==0: return (0.0,0.0,0.3)
    if idx==1: return (0.2,0.4,0.6)
    if idx==2: return (0.5,0.7,0.9)
    return (0.8,1.0,1.0)

class FuzzyRule:
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent

def define_fuzzy_rules():
    rules = []
    rules.append(FuzzyRule(antecedent=[(0,2),(1,0)], consequent=3))
    rules.append(FuzzyRule(antecedent=[(0,1),(1,1),(2,0)], consequent=1))
    rules.append(FuzzyRule(antecedent=[(0,0),(1,2)], consequent=1))
    rules.append(FuzzyRule(antecedent=[(0,0),(1,0)], consequent=0))
    rules.append(FuzzyRule(antecedent=[(0,2),(2,2)], consequent=2))
    return rules

def fuzzy_inference_single_sample(fv, fuzzy_sets, rules):
    memberships = {i:[] for i in range(cfg.num_fuzzy_features)}
    for feat_idx in range(cfg.num_fuzzy_features):
        x = float(fv[feat_idx])
        for (a,b,c) in fuzzy_sets[feat_idx]:
            memberships[feat_idx].append(tri_membership(x,a,b,c))
    rule_strengths=[]
    rule_outputs=[]
    for r in rules:
        mus = [memberships[f][s] for f,s in r.antecedent]
        strength = min(mus) if mus else 0.0
        rule_strengths.append(strength); rule_outputs.append(r.consequent)
    R=101; zvals = np.linspace(0.0,1.0,R); agg_mu = np.zeros_like(zvals)
    for r_idx, strength in enumerate(rule_strengths):
        cons = rule_outputs[r_idx]; a,b,c = risk_output_fuzzy_set(cons)
        for i,z in enumerate(zvals):
            muo = tri_membership(z,a,b,c); agg_mu[i] = max(agg_mu[i], min(strength, muo))
    if agg_mu.sum() < 1e-8: return 0.0
    crisp = (zvals * agg_mu).sum() / (agg_mu.sum() + 1e-8)
    return float(crisp)

def normalize_fuzzy_features(fv):
    return np.clip(fv, 0.0, 1.0)

def compute_fuzzy_consistency(model, val_loader, fuzzy_params):
    model.eval()
    fuzzy_sets = build_fuzzy_membership_functions(fuzzy_params)
    rules = define_fuzzy_rules()
    labels_all=[]; risks_all=[]
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(cfg.device); labels = labels.to(cfg.device)
            logits, feats = model(imgs)
            probs = F.softmax(logits, dim=1)
            severity = probs[:,-1]; ent = entropy_from_probs(probs)
            vessel = compute_vessel_density(imgs); texture = compute_texture_score(feats)
            for i in range(imgs.size(0)):
                fv = np.array([
                    float(severity[i].cpu()),
                    float(ent[i].cpu() / math.log(cfg.num_classes + 1e-8)),
                    float(vessel[i].cpu()),
                    float(texture[i].cpu()),
                ], dtype=np.float32)
                fv_norm = normalize_fuzzy_features(fv)
                crisp = fuzzy_inference_single_sample(fv_norm, fuzzy_sets, rules)
                labels_all.append(int(labels[i].cpu())); risks_all.append(crisp)
    if len(set(labels_all)) <= 1: return 0.0
    corr, _ = spearmanr(labels_all, risks_all)
    if corr is None or math.isnan(corr): return 0.0
    return float((corr + 1.0)/2.0)

# ---------------------------
# GA evaluation and operators
# ---------------------------
def evaluate_chromosome(chrom, train_samples, val_samples):
    train_tfms, val_tfms = build_transforms(aug_intensity=chrom.aug_intensity if hasattr(chrom, 'aug_intensity') else 0.6)
    num_workers = 0 if os.name == 'nt' else 4
    train_ds = DRDataset(train_samples, transform=train_tfms)
    val_ds = DRDataset(val_samples, transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=chrom.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=chrom.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model = NeuroBackbone(num_classes=cfg.num_classes, feature_dim=cfg.feature_dim, dropout=chrom.dropout, pretrained=True)
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=chrom.lr, weight_decay=chrom.weight_decay)
    criterion = nn.CrossEntropyLoss()
    # new amp API
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    best_kappa = float('-inf')
    for epoch in range(1, cfg.epochs_ga_eval + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(cfg.device); labels = labels.to(cfg.device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=cfg.use_amp):
                logits, _ = model(imgs); loss = criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        acc, kappa = eval_model(model, val_loader, cfg.device)
        best_kappa = max(best_kappa, kappa)
    fuzzy_cons = compute_fuzzy_consistency(model, val_loader, chrom.fuzzy_params)
    chrom.fitness = best_kappa + 0.1 * fuzzy_cons
    log(f"[GA] fitness={chrom.fitness:.4f}, kappa={best_kappa:.4f}, fuzzy={fuzzy_cons:.4f}")
    return chrom.fitness

def tournament_selection(pop, k=3):
    cand = random.choice(pop)
    for _ in range(k-1):
        challenger = random.choice(pop)
        if challenger.fitness > cand.fitness:
            cand = challenger
    newc = Chromosome(); newc.lr = cand.lr; newc.weight_decay = cand.weight_decay; newc.batch_size = cand.batch_size
    newc.dropout = cand.dropout; newc.aug_intensity = cand.aug_intensity; newc.fuzzy_params = cand.fuzzy_params.copy(); newc.fitness = cand.fitness
    return newc

def mutate(chrom):
    if random.random() < cfg.ga_mutation_rate:
        chrom.lr *= math.exp(random.gauss(0, 0.3)); chrom.lr = float(np.clip(chrom.lr, 1e-6, 1e-2))
    if random.random() < cfg.ga_mutation_rate:
        chrom.weight_decay *= math.exp(random.gauss(0, 0.3)); chrom.weight_decay = float(np.clip(chrom.weight_decay, 1e-7, 1e-3))
    if random.random() < cfg.ga_mutation_rate:
        chrom.batch_size = random.choice([16,24,32,40])
    if random.random() < cfg.ga_mutation_rate:
        chrom.dropout += random.gauss(0,0.05); chrom.dropout = float(np.clip(chrom.dropout, 0.05, 0.7))
    for i in range(len(chrom.fuzzy_params)):
        if random.random() < (cfg.ga_mutation_rate/2.0):
            chrom.fuzzy_params[i] += random.gauss(0,0.03); chrom.fuzzy_params[i] = float(np.clip(chrom.fuzzy_params[i], 0.0, 1.0))
    chrom.fitness = float('-inf'); return chrom

def crossover(p1, p2):
    c1 = Chromosome(); c2 = Chromosome()
    if random.random() < cfg.ga_crossover_rate:
        genes1 = [p1.lr, p1.weight_decay, float(p1.batch_size), p1.dropout, p1.aug_intensity]
        genes2 = [p2.lr, p2.weight_decay, float(p2.batch_size), p2.dropout, p2.aug_intensity]
        point = random.randint(1, len(genes1)-1)
        newg1 = genes1[:point] + genes2[point:]; newg2 = genes2[:point] + genes1[point:]
    else:
        newg1 = [p1.lr, p1.weight_decay, float(p1.batch_size), p1.dropout, p1.aug_intensity]
        newg2 = [p2.lr, p2.weight_decay, float(p2.batch_size), p2.dropout, p2.aug_intensity]
    c1.lr, c1.weight_decay, bs1, c1.dropout, c1.aug_intensity = newg1
    c2.lr, c2.weight_decay, bs2, c2.dropout, c2.aug_intensity = newg2
    c1.batch_size = int(bs1); c2.batch_size = int(bs2)
    c1.fuzzy_params = []; c2.fuzzy_params = []
    for a,b in zip(p1.fuzzy_params, p2.fuzzy_params):
        if random.random() < 0.5:
            c1.fuzzy_params.append(a); c2.fuzzy_params.append(b)
        else:
            c1.fuzzy_params.append(b); c2.fuzzy_params.append(a)
    c1.fitness = float('-inf'); c2.fitness = float('-inf')
    return c1,c2

def run_ga(train_samples, val_samples):
    log('=== GA START ==='); pop = init_population(cfg.ga_population); best=None
    for gen in range(1, cfg.ga_generations+1):
        log(f'[GA] Generation {gen}/{cfg.ga_generations}')
        for chrom in pop:
            if chrom.fitness == float('-inf'):
                evaluate_chromosome(chrom, train_samples, val_samples)
        pop.sort(key=lambda x: x.fitness, reverse=True)
        if best is None or pop[0].fitness > best.fitness:
            best = pop[0]; log(f'[GA] New best fitness: {best.fitness:.4f}'); save_ga_state(best)
        # elitism
        newpop = []
        for i in range(cfg.ga_elitism_k):
            elite = pop[i]; c = Chromosome(); c.lr=elite.lr; c.weight_decay=elite.weight_decay; c.batch_size=elite.batch_size
            c.dropout=elite.dropout; c.aug_intensity=elite.aug_intensity; c.fuzzy_params=elite.fuzzy_params.copy(); c.fitness=elite.fitness
            newpop.append(c)
        while len(newpop) < cfg.ga_population:
            p1 = tournament_selection(pop); p2 = tournament_selection(pop)
            c1, c2 = crossover(p1,p2); c1 = mutate(c1); c2 = mutate(c2)
            newpop.append(c1)
            if len(newpop) < cfg.ga_population: newpop.append(c2)
        pop = newpop
    log('=== GA END ==='); return best

# GA state save/load
def save_ga_state(chrom):
    ensure_dirs(os.path.join(cfg.work_dir, cfg.ckpt_dir))
    path = os.path.join(cfg.work_dir, cfg.ckpt_dir, 'best_ga_state.json')
    with open(path, 'w') as f: json.dump(chrom.to_dict(), f, indent=2)
    log(f'[GA] Saved GA state -> {path}')

def load_ga_state(path):
    if not os.path.isfile(path): return None
    with open(path,'r') as f: d = json.load(f)
    return Chromosome.from_dict(d)

# ---------------------------
# Checkpointing (multiple files)
# ---------------------------
def list_checkpoints(resume_dir=None):
    if resume_dir is None: resume_dir = os.path.join(cfg.work_dir, cfg.ckpt_dir)
    if not os.path.isdir(resume_dir): return []
    files = [f for f in os.listdir(resume_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    files.sort()
    return files

def save_checkpoint(epoch, model, optimizer, scaler):
    ensure_dirs(os.path.join(cfg.work_dir, cfg.ckpt_dir))
    path = os.path.join(cfg.work_dir, cfg.ckpt_dir, f'checkpoint_epoch_{epoch:02d}.pt')
    data = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict() if scaler else None
    }
    torch.save(data, path)
    log(f'[CHECKPOINT] Saved {path}')

def load_latest_checkpoint(model, optimizer=None, scaler=None):
    ckpts = list_checkpoints()
    if not ckpts: return None
    latest = ckpts[-1]
    path = os.path.join(cfg.work_dir, cfg.ckpt_dir, latest)
    log(f'[RESUME] Loading checkpoint: {latest}')
    d = torch.load(path, map_location=cfg.device)
    model.load_state_dict(d['model_state'])
    if optimizer is not None and 'optimizer_state' in d and d['optimizer_state'] is not None:
        optimizer.load_state_dict(d['optimizer_state'])
    if scaler is not None and 'scaler_state' in d and d['scaler_state'] is not None:
        scaler.load_state_dict(d['scaler_state'])
    return d.get('epoch', None)

# ---------------------------
# Training full model
# ---------------------------
def train_full_model(chrom, all_train_samples, writer=None, resume=True):
    log('=== FULL TRAINING START ===')
    train_samples, val_samples = train_test_split(all_train_samples, test_size=cfg.val_ratio, stratify=[l for _,l in all_train_samples], random_state=cfg.seed)
    train_tfms, val_tfms = build_transforms()
    num_workers = 0 if os.name=='nt' else 6
    train_ds = DRDataset(train_samples, transform=train_tfms)
    val_ds = DRDataset(val_samples, transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=chrom.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=chrom.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model = NeuroBackbone(num_classes=cfg.num_classes, feature_dim=cfg.feature_dim, dropout=chrom.dropout, pretrained=True)
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=chrom.lr, weight_decay=chrom.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs_full_train))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 1
    best_kappa = float('-inf')
    # resume
    if resume:
        ep = load_latest_checkpoint(model, optimizer, scaler)
        if ep is not None:
            start_epoch = ep + 1
            log(f'[RESUME] Resuming from epoch {ep} -> starting at {start_epoch}')
    # training loop
    for epoch in range(start_epoch, cfg.epochs_full_train + 1):
        model.train()
        running = 0.0; n = 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(cfg.device); labels = labels.to(cfg.device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=cfg.use_amp):
                logits, _ = model(imgs); loss = criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item() * imgs.size(0); n += imgs.size(0)
        train_loss = running / max(1, n)
        val_acc, val_kappa = eval_model(model, val_loader, cfg.device)
        scheduler.step()
        elapsed = time.time() - t0
        log(f'[EPOCH {epoch}/{cfg.epochs_full_train}] train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_kappa={val_kappa:.4f} time={elapsed:.1f}s')
        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch); writer.add_scalar('val/kappa', val_kappa, epoch)
        # save checkpoint every epoch
        save_checkpoint(epoch, model, optimizer, scaler)
        # save best model copy by kappa
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            ensure_dirs(os.path.join(cfg.work_dir, cfg.ckpt_dir))
            torch.save(model.state_dict(), os.path.join(cfg.work_dir, cfg.best_model_path))
            log(f'[EPOCH {epoch}] New best model saved (kappa={best_kappa:.4f})')
    # load best model final
    best_model = NeuroBackbone(num_classes=cfg.num_classes, feature_dim=cfg.feature_dim, dropout=chrom.dropout, pretrained=True)
    best_model.load_state_dict(torch.load(os.path.join(cfg.work_dir, cfg.best_model_path), map_location=cfg.device))
    best_model.to(cfg.device)
    return best_model

# ---------------------------
# Evaluate on test with fuzzy combo
# ---------------------------
def evaluate_on_test(model, chrom, test_samples):
    _, test_tfms = build_transforms()
    num_workers = 0 if os.name=='nt' else 6
    test_ds = DRDataset(test_samples, transform=test_tfms)
    test_loader = DataLoader(test_ds, batch_size=chrom.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    fuzzy_sets = build_fuzzy_membership_functions(chrom.fuzzy_params); rules = define_fuzzy_rules()
    all_true=[]; all_base=[]; all_final=[]
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(cfg.device); labels = labels.to(cfg.device)
            logits, feats = model(imgs); probs = F.softmax(logits, dim=1)
            base_preds = torch.argmax(probs, dim=1)
            severity = probs[:,-1]; ent = entropy_from_probs(probs); vessel = compute_vessel_density(imgs); texture = compute_texture_score(feats)
            for i in range(imgs.size(0)):
                fv = np.array([float(severity[i].cpu()), float(ent[i].cpu() / math.log(cfg.num_classes + 1e-8)), float(vessel[i].cpu()), float(texture[i].cpu())], dtype=np.float32)
                fv_norm = normalize_fuzzy_features(fv); risk = fuzzy_inference_single_sample(fv_norm, fuzzy_sets, rules)
                base_c = int(base_preds[i].cpu()); final_c = base_c
                if risk > 0.8 and base_c < cfg.num_classes - 1: final_c = base_c + 1
                if risk < 0.2 and base_c > 0: final_c = base_c - 1
                all_true.append(int(labels[i].cpu())); all_base.append(base_c); all_final.append(final_c)
    base_acc = accuracy_score(all_true, all_base); final_acc = accuracy_score(all_true, all_final)
    base_kappa = compute_kappa(all_true, all_base); final_kappa = compute_kappa(all_true, all_final)
    log('=== TEST RESULTS ==='); log(f'Base  model: acc={base_acc:.4f}, kappa={base_kappa:.4f}'); log(f'Final model: acc={final_acc:.4f}, kappa={final_kappa:.4f}'); log(classification_report(all_true, all_final, digits=4))

# ---------------------------
# Main orchestration
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=str, default=os.getcwd())
    parser.add_argument('--train-csv', type=str, help='train csv filename relative to work-dir')
    parser.add_argument('--val-csv', type=str, help='val csv filename relative to work-dir')
    parser.add_argument('--test-csv', type=str, help='test csv filename relative to work-dir')
    parser.add_argument('--epochs', type=int, help='full training epochs')
    parser.add_argument('--no-ga', action='store_true', help='skip GA and load saved GA state if available')
    args = parser.parse_args()

    cfg.work_dir = args.work_dir
    if args.train_csv: cfg.train_csv = args.train_csv
    if args.val_csv: cfg.val_csv = args.val_csv
    if args.test_csv: cfg.test_csv = args.test_csv
    if args.epochs: cfg.epochs_full_train = args.epochs

    set_seed(cfg.seed)
    ensure_dirs(os.path.join(cfg.work_dir, cfg.ckpt_dir))
    print_cuda_banner()

    # check cuda availability and warn
    if cfg.device == 'cpu':
        log('[WARN] Running on CPU. Install CUDA-enabled PyTorch to use GPU.', True)

    train_csv_path = os.path.join(cfg.work_dir, cfg.train_csv)
    val_csv_path = os.path.join(cfg.work_dir, cfg.val_csv)
    test_csv_path = os.path.join(cfg.work_dir, cfg.test_csv)
    train_img_dir = os.path.join(cfg.work_dir, cfg.train_img_dir)
    val_img_dir = os.path.join(cfg.work_dir, cfg.val_img_dir)
    test_img_dir = os.path.join(cfg.work_dir, cfg.test_img_dir)

    log(f'Work dir: {cfg.work_dir}'); log(f'Train CSV: {train_csv_path}')
    train_pairs = load_csv_pairs(train_csv_path, train_img_dir)
    val_pairs = load_csv_pairs(val_csv_path, val_img_dir)
    test_pairs = load_csv_pairs(test_csv_path, test_img_dir)

    if not val_pairs:
        log('[INFO] No separate validation CSV; splitting train into train/val')
        train_pairs, val_pairs = train_test_split(train_pairs, test_size=cfg.val_ratio, stratify=[l for _,l in train_pairs], random_state=cfg.seed)
    log(f'Load counts -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}')

    # GA
    best_ga_path = os.path.join(cfg.work_dir, cfg.ckpt_dir, 'best_ga_state.json')
    best_chrom = None
    if not args.no_ga:
        if os.path.isfile(best_ga_path):
            try:
                best_chrom = load_ga_state(best_ga_path)
                log(f'[GA] Loaded GA state with fitness: {best_chrom.fitness:.4f}')
            except Exception:
                best_chrom = run_ga(train_pairs, val_pairs)
        else:
            best_chrom = run_ga(train_pairs, val_pairs)
    else:
        log('[GA] Skipping GA as requested (using defaults)')

    if best_chrom is None:
        best_chrom = init_random_chromosome()
        log('[INFO] Using random chromosome defaults')

    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'runs'))

    # full training (resume-enabled)
    best_model = train_full_model(best_chrom, train_pairs + val_pairs, writer, resume=True)

    # evaluation on test
    evaluate_on_test(best_model, best_chrom, test_pairs)

    writer.close()
    log('[DONE] All finished.')

if __name__ == '__main__':
    main()