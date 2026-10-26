# train_models.py
"""Train one of the three feature-group LSTM autoencoders.

    python 2_training/train_models.py --view position --seed 42
    python 2_training/train_models.py --view speed    --seed 42
    python 2_training/train_models.py --view vertical --seed 42

The configurations used in the paper are defined in CONFIGS below
(position: E_pure_mse, speed: I_latent_smoothing,
vertical: I_altitude_latent_smoothing).
"""
import os as _os, sys as _sys  # REPO_ROOT_SHIM
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

_BASE = os.environ.get("ADSB_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES = {
    "position": ("dx", "dy", "turn_intensity"),
    "speed": ("groundspeed", "delta_gs", "gs_diff"),
    "vertical": ("altitude", "vertical_rate", "vs_diff"),
}

CONFIGS = {
    "position": dict(
        name="E_pure_mse", bidirectional=True,
        encoder_units=[96, 48], decoder_units=[96, 192],
        latent_dim=24, latent_activation="tanh",
        optimizer="rmsprop", lr=1e-3,
        losses={"dx_out": "mse", "dy_out": "mse", "turn_intensity_out": "mse"},
    ),
    "speed": dict(
        name="I_latent_smoothing", bidirectional=True,
        encoder_units=[128, 64], decoder_units=[128, 256],
        latent_dim=32, latent_activation="relu",
        optimizer="adam", lr=5e-4,
        losses={"groundspeed_out": "huber", "delta_gs_out": "huber",
                "gs_diff_out": "huber"},
        loss_weights={"groundspeed_out": 3.0, "delta_gs_out": 1.0,
                      "gs_diff_out": 1.0},
        noise_std=0.003, dropout=0.05,
        use_latent_layernorm=True,
    ),
    "vertical": dict(
        name="I_altitude_latent_smoothing", bidirectional=True,
        encoder_units=[128, 64], decoder_units=[128, 256],
        latent_dim=32, latent_activation="relu",
        optimizer="adam", lr=5e-4,
        losses={"altitude_out": "huber", "vertical_rate_out": "huber",
                "vs_diff_out": "huber"},
        loss_weights={"altitude_out": 3.0, "vertical_rate_out": 1.0,
                      "vs_diff_out": 1.0},
        noise_std=0.003, dropout=0.05,
        use_latent_layernorm=True,
    ),
}


def make_targets(X_scaled, feature_names):
    """One target per feature, matching the output heads `<feature>_out`."""
    return {f"{name}_out": X_scaled[:, :, i:i + 1] for i, name in enumerate(feature_names)}


def build_model(config, feature_names):
    """Builds and compiles the LSTM autoencoder (encoder, latent layer, decoder, one output head per feature)."""
    inputs = layers.Input(shape=(60, len(feature_names)), name="Input")
    x = inputs

    # Encoder
    for i, units in enumerate(config["encoder_units"]):
        layer = layers.LSTM(units, return_sequences=(i < len(config["encoder_units"]) - 1))
        if config["bidirectional"]:
            layer = layers.Bidirectional(layer)
        x = layer(x)
        x = layers.LayerNormalization()(x)

    latent = layers.Dense(config["latent_dim"], activation=config["latent_activation"])(x)
    latent = layers.GaussianNoise(config.get("noise_std", 0.01))(latent)
    latent = layers.Dropout(config.get("dropout", 0.1))(latent)
    if config.get("use_latent_layernorm", False):
        latent = layers.LayerNormalization()(latent)

    # Decoder
    x = layers.RepeatVector(60)(latent)
    for units in config["decoder_units"]:
        x = layers.LSTM(units, return_sequences=True)(x)
        x = layers.LayerNormalization()(x)

    # One output head per feature
    outputs = [layers.TimeDistributed(layers.Dense(1), name=f"{feat}_out")(x)
               for feat in feature_names]
    model = models.Model(inputs, outputs, name=config["name"])

    optimizer_map = {
        "adam": tf.keras.optimizers.Adam(config["lr"]),
        "rmsprop": tf.keras.optimizers.RMSprop(config["lr"]),
        "nadam": tf.keras.optimizers.Nadam(config["lr"]),
    }
    loss_map = {"mse": "mse", "huber": tf.keras.losses.Huber(delta=1.0)}
    losses = {k: loss_map[v] for k, v in config["losses"].items()}
    default_weights = {f"{f}_out": 1.0 for f in feature_names}
    loss_weights = config.get("loss_weights", default_weights)

    model.compile(optimizer=optimizer_map[config["optimizer"]], loss=losses,
                  loss_weights=loss_weights)
    return model


def set_seed(seed, view):
    """Seeding exactly as in the runs that produced the paper models.

    The position model was trained with the explicit seed calls and
    TF_CUDNN_DETERMINISTIC; speed and vertical with keras' set_random_seed.
    The two are kept apart on purpose: with Keras 3 they can initialise the
    weights differently, so unifying them would change the trained models.
    """
    if view == "position":
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    else:
        tf.keras.utils.set_random_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"


def main():
    """Command line entry point: loads the scaled sequences of one view and trains one model for one seed."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True, choices=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default=None,
                        help="default: $ADSB_BASE/artifacts/sequences/<view>")
    parser.add_argument("--out_dir", default=None,
                        help="default: $ADSB_BASE/artifacts/model_<view>")
    args = parser.parse_args()

    view = args.view
    feature_names = FEATURES[view]
    config = CONFIGS[view]
    data_dir = args.data_dir or os.path.join(_BASE, "artifacts/sequences", view)
    out_dir = args.out_dir or os.path.join(_BASE, f"artifacts/model_{view}")

    set_seed(args.seed, view)
    os.makedirs(out_dir, exist_ok=True)
    name_seed = f"{config['name']}_seed{args.seed}"

    npz = f"{data_dir}/sequences.npz"
    if not os.path.exists(npz):
        raise SystemExit(f"Missing {npz}. Run `python 2_training/make_views.py` first "
                         "(see README, 'Reproducing the paper').")
    data = np.load(npz)
    X_train_scaled = data["X_train_scaled"]
    X_val_scaled = data["X_val_scaled"]
    assert X_train_scaled.shape[2] == len(feature_names), \
        f"Expected {len(feature_names)} features, found {X_train_scaled.shape[2]}"

    model = build_model(config, feature_names)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                                             min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(f"{out_dir}/{name_seed}.keras",
                                           monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(f"{out_dir}/{name_seed}_log.csv"),
    ]

    history = model.fit(
        X_train_scaled, make_targets(X_train_scaled, feature_names),
        validation_data=(X_val_scaled, make_targets(X_val_scaled, feature_names)),
        epochs=150, batch_size=256, shuffle=True, callbacks=callbacks, verbose=2,
    )

    with open(f"{out_dir}/{name_seed}_history.json", "w") as f:
        json.dump(history.history, f, indent=2)


if __name__ == "__main__":
    main()