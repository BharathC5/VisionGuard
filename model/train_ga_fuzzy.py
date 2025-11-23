# stage2_fuzzy_inference.py

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_PATH = "best_multitask_aptos.h5"
TRAIN_DIR  = "data_pp/train"
TEST_DIR   = "data_pp/test"   # preprocessed test images
IMG_SIZE   = (300, 300)
NUM_CLASSES = 5
CLASS_IDS  = [0, 1, 2, 3, 4]
SEED = 42

# GA config
POP_SIZE = 8
N_GENERATIONS = 3
FITNESS_EPOCHS = 3
ELITISM_K = 3
MUTATION_RATE = 0.3

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------- FUZZY HELPERS (OUTPUT) ------------

def triangular_mu(sev, k):
    """
    Triangular membership with base [k-1, k+1], peak at k.
    sev: severity score (0-4), k: stage index (0-4)
    """
    if sev <= k - 1 or sev >= k + 1:
        return 0.0
    # base width is 2, max height 1
    return 1.0 - abs(sev - k)

def fuzzy_output(probs, sev, alpha=0.5):
    """
    Combine softmax probs + severity into fuzzy memberships.
    probs: array shape (5,)
    sev: scalar severity (0-4)
    alpha: weight for probs vs severity memberships
    """
    probs = np.array(probs, dtype=np.float32)
    severity = float(sev)

    mu_prob = probs
    mu_sev  = np.array([triangular_mu(severity, k) for k in range(len(probs))],
                       dtype=np.float32)
    mu_final = alpha * mu_prob + (1.0 - alpha) * mu_sev

    order = np.argsort(mu_final)[::-1]
    top1, top2 = int(order[0]), int(order[1])

    diff = float(mu_final[top1] - mu_final[top2])
    if diff > 0.30:
        ambiguity = "confident"
    elif diff > 0.12:
        ambiguity = "borderline"
    else:
        ambiguity = "very_ambiguous"

    center = top1
    if severity < center - 0.25:
        timing = "early"
    elif severity > center + 0.25:
        timing = "late"
    else:
        timing = "mid"

    return {
        "top_stage": top1,
        "top_mu": float(mu_final[top1]),
        "second_stage": top2,
        "second_mu": float(mu_final[top2]),
        "severity": round(severity, 3),
        "ambiguity": ambiguity,
        "timing": timing,
    }

# --------- DATA LOADING (TRAIN/VAL/TEST) ---------

def make_train_val_ds(batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    # Map single label -> two heads: cls (int), sev (float)
    def map_targets(images, labels):
        labels_float = tf.cast(labels, tf.float32)
        return images, {"cls": labels, "sev": labels_float}

    return train_ds.map(map_targets), val_ds.map(map_targets)

def make_full_train_ds(batch_size):
    ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size
    )
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.shuffle(1000).prefetch(AUTOTUNE)

    def map_targets(images, labels):
        labels_float = tf.cast(labels, tf.float32)
        return images, {"cls": labels, "sev": labels_float}

    return ds.map(map_targets)

def list_test_images(root):
    paths = []
    labels = []
    for cls in CLASS_IDS:
        cls_dir = os.path.join(root, str(cls))
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ):
                continue
            paths.append(os.path.join(cls_dir, fname))
            labels.append(cls)
    return paths, labels

def load_and_preprocess_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    return arr

# ---------------- MULTI-TASK MODEL ----------------

def build_multitask_model(dropout=0.4, unfreeze_layers=40,
                          label_smoothing=0.1,
                          lr=1e-4, lambda_cls=1.0, lambda_reg=1.0):
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomContrast(0.15)(x)
    x = preprocess_input(x)

    base = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")
    base.trainable = True
    # fuzzy-style unfreezing: only last N layers are trainable
    for layer in base.layers[:-unfreeze_layers]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    # Head 1: classification softmax
    cls_out = layers.Dense(NUM_CLASSES, activation="softmax", name="cls")(x)
    # Head 2: severity regression
    sev_out = layers.Dense(1, activation="linear", name="sev")(x)

    model = models.Model(inputs, [cls_out, sev_out])

    cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        label_smoothing=label_smoothing
    )
    reg_loss = tf.keras.losses.MeanSquaredError()

    losses = {"cls": cls_loss, "sev": reg_loss}
    loss_weights = {"cls": lambda_cls, "sev": lambda_reg}

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={"cls": "accuracy"}
    )
    return model

# ---------------- GA: HYPERPARAMS ----------------

def random_individual():
    return {
        "lr":              10 ** random.uniform(-5, -3),
        "dropout":         random.uniform(0.2, 0.6),
        "unfreeze_layers": random.randint(10, 80),
        "label_smoothing": random.uniform(0.0, 0.2),
        "lambda_cls":      random.uniform(0.5, 1.5),
        "lambda_reg":      random.uniform(0.5, 1.5),
        "batch_size":      random.choice([8, 16, 32])
    }

def crossover(p1, p2):
    child = {}
    for k in p1.keys():
        child[k] = random.choice([p1[k], p2[k]])
    return child

def mutate(ind):
    if random.random() < MUTATION_RATE:
        ind["lr"] = 10 ** random.uniform(-5, -3)
    if random.random() < MUTATION_RATE:
        ind["dropout"] = min(0.7, max(0.1, ind["dropout"] + random.uniform(-0.1, 0.1)))
    if random.random() < MUTATION_RATE:
        ind["unfreeze_layers"] = max(10, min(120, ind["unfreeze_layers"] +
                                             random.randint(-10, 10)))
    if random.random() < MUTATION_RATE:
        ind["label_smoothing"] = min(
            0.3, max(0.0, ind["label_smoothing"] + random.uniform(-0.05, 0.05))
        )
    if random.random() < MUTATION_RATE:
        ind["lambda_cls"] = min(
            2.0, max(0.1, ind["lambda_cls"] + random.uniform(-0.3, 0.3))
        )
    if random.random() < MUTATION_RATE:
        ind["lambda_reg"] = min(
            2.0, max(0.1, ind["lambda_reg"] + random.uniform(-0.3, 0.3))
        )
    if random.random() < MUTATION_RATE:
        ind["batch_size"] = random.choice([8, 16, 32])
    return ind

def evaluate_fitness(ind):
    print("\nEvaluating individual:", ind)
    bs = ind["batch_size"]
    train_ds, val_ds = make_train_val_ds(batch_size=bs)

    model = build_multitask_model(
        dropout=ind["dropout"],
        unfreeze_layers=ind["unfreeze_layers"],
        label_smoothing=ind["label_smoothing"],
        lr=ind["lr"],
        lambda_cls=ind["lambda_cls"],
        lambda_reg=ind["lambda_reg"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FITNESS_EPOCHS,
        verbose=0
    )
    val_acc = history.history["val_cls_accuracy"][-1]
    print(f"  -> val_cls_accuracy = {val_acc:.4f}")
    return val_acc

def run_ga_and_train_final():
    print("Running GA to find good hyperparameters...")
    population = [random_individual() for _ in range(POP_SIZE)]
    fitness_cache = {}

    for gen in range(N_GENERATIONS):
        print(f"\n===== Generation {gen+1}/{N_GENERATIONS} =====")
        scored = []
        for i, ind in enumerate(population):
            key = tuple(sorted(ind.items()))
            if key in fitness_cache:
                fit = fitness_cache[key]
                print(f"Using cached fitness for individual {i}")
            else:
                fit = evaluate_fitness(ind)
                fitness_cache[key] = fit
            scored.append((ind, fit))

        scored.sort(key=lambda x: x[1], reverse=True)

        print("Top this generation:")
        for rank, (ind, fit) in enumerate(scored[:3]):
            print(f"  {rank+1}. fit={fit:.4f}, ind={ind}")

        new_pop = [ind for (ind, _) in scored[:ELITISM_K]]
        while len(new_pop) < POP_SIZE:
            p1, _ = random.choice(scored[:5])
            p2, _ = random.choice(scored[:5])
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

    # Choose best overall
    final_scored = []
    for ind in population:
        key = tuple(sorted(ind.items()))
        fit = fitness_cache.get(key, evaluate_fitness(ind))
        final_scored.append((ind, fit))
    final_scored.sort(key=lambda x: x[1], reverse=True)
    best_ind, best_fit = final_scored[0]

    print("\n===== GA DONE =====")
    print("Best individual:", best_ind)
    print(f"Best val_cls_accuracy: {best_fit:.4f}")

    # Final training on full train set with best hyperparams
    print("\nTraining final model with best hyperparameters on full train set...")
    bs = best_ind["batch_size"]
    full_train_ds = make_full_train_ds(batch_size=bs)

    model = build_multitask_model(
        dropout=best_ind["dropout"],
        unfreeze_layers=best_ind["unfreeze_layers"],
        label_smoothing=best_ind["label_smoothing"],
        lr=best_ind["lr"],
        lambda_cls=best_ind["lambda_cls"],
        lambda_reg=best_ind["lambda_reg"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    model.fit(
        full_train_ds,
        epochs=20,    # you can increase if GPU allows
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")
    return best_ind

# ---------------- MAIN: GA TRAIN + FUZZY INFERENCE ----------------

def run_fuzzy_inference():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Listing test images...")
    img_paths, true_labels = list_test_images(TEST_DIR)
    true_labels = np.array(true_labels, dtype=np.int32)

    all_preds = []
    all_sev   = []
    all_fuzzy = []

    correct = 0

    print(f"Found {len(img_paths)} test images.")

    for i, path in enumerate(img_paths):
        arr = load_and_preprocess_image(path)
        batch = np.expand_dims(arr, axis=0)

        preds = model.predict(batch, verbose=0)
        if isinstance(preds, (list, tuple)):
            probs = preds[0][0]        # (5,)
            sev   = float(preds[1][0]) # scalar
        else:
            probs = preds[0]
            sev   = float(np.sum(np.arange(len(probs)) * probs))

        fuzzy = fuzzy_output(probs, sev, alpha=0.5)

        pred_class = int(np.argmax(probs))
        if pred_class == true_labels[i]:
            correct += 1

        all_preds.append(pred_class)
        all_sev.append(sev)
        all_fuzzy.append(fuzzy)

        if (i + 1) % 50 == 0 or i == len(img_paths) - 1:
            print(f"Processed {i+1}/{len(img_paths)} images...")

    test_acc = correct / len(img_paths) * 100.0
    print(f"\nTest accuracy (hard argmax): {test_acc:.2f}%")

    # -------- save detailed fuzzy report --------
    records = []
    for path, y_true, y_pred, sev, fz in zip(
        img_paths, true_labels, all_preds, all_sev, all_fuzzy
    ):
        rec = {
            "image_path": path,
            "true_label": int(y_true),
            "pred_label": int(y_pred),
            "severity": fz["severity"],
            "top_stage": fz["top_stage"],
            "top_mu": fz["top_mu"],
            "second_stage": fz["second_stage"],
            "second_mu": fz["second_mu"],
            "ambiguity": fz["ambiguity"],
            "timing": fz["timing"],  # early / mid / late
        }
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv("stage2_fuzzy_results.csv", index=False)
    print("Saved detailed fuzzy results to stage2_fuzzy_results.csv")

    print("\nSample fuzzy outputs:")
    print(df.head().to_string(index=False))

def main():
    # If model does not exist yet, run GA + final train; else reuse
    if not os.path.exists(MODEL_PATH):
        run_ga_and_train_final()
    run_fuzzy_inference()

if __name__ == "__main__":
    main()
