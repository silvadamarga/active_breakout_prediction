# extract_hps.py
import keras_tuner as kt
import json
import os
import numpy as np
import tensorflow as tf # Import tensorflow directly
import logging # Added for clearer output

# --- Configuration (Match your main script) ---
CACHE_DIR = "data_cache_combined"
BEST_HPS_CACHE = os.path.join(CACHE_DIR, "checkpoint_best_hps.json")
TUNER_PROJECT_NAME = 'breakout_v5_combined' # Should match project_name in main script
# --- End Configuration ---

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info(f"Attempting to load tuner state from directory: keras_tuner_dir/{TUNER_PROJECT_NAME}")

# --- Dummy build function (needed to reload tuner) ---
# --- IMPORTANT: Make sure TIME_STEPS and N_FEATURES match your run ---
TIME_STEPS = 60
N_FEATURES = 19 # From your log: "Using 19 features for breakout model."

def dummy_build_model(hp):
    # Use tf.keras instead of kt.keras
    inp = tf.keras.Input(shape=(TIME_STEPS, N_FEATURES))
    # Need at least one layer to compile
    x = tf.keras.layers.Flatten()(inp) # Flatten to ensure output shape is simple
    out = tf.keras.layers.Dense(1)(x) # Minimal layer
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse') # Compile is necessary
    return model
# --- End Dummy build function ---

try:
    # Reload the tuner object from the directory
    tuner = kt.Hyperband(
        dummy_build_model, # Use the dummy function
        objective=kt.Objective("val_breakout_auc", direction="max"), # Must match objective
        max_epochs=15, # Must match max_epochs from main script
        factor=3,      # Must match factor from main script
        directory='keras_tuner_dir',
        project_name=TUNER_PROJECT_NAME,
        overwrite=False # IMPORTANT: Set to False to load existing state
    )

    logging.info("Tuner loaded. Fetching best hyperparameters...")

    # Check if any hyperparameters were found
    best_hps_list = tuner.get_best_hyperparameters(num_trials=1)

    if not best_hps_list:
        logging.error("Keras Tuner did not find any completed trials with valid scores in its cache.")
        logging.error("This might indicate corrupted cache files or that no trials finished successfully.")
        logging.error("You may need to delete 'keras_tuner_dir' and re-run the main script to start the tuner search again.")
    else:
        # Get the best hyperparameters found
        best_hps_raw = best_hps_list[0].values
        logging.info("Successfully retrieved best hyperparameters from tuner cache:")
        logging.info(best_hps_raw)

        # Clean HPs for JSON serialization (convert numpy types)
        best_hps_cleaned = {
            k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
            for k, v in best_hps_raw.items()
        }

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Save the cleaned HPs to the checkpoint file
        with open(BEST_HPS_CACHE, 'w') as f:
            json.dump(best_hps_cleaned, f, indent=4)

        logging.info(f"Successfully saved best hyperparameters to: {BEST_HPS_CACHE}")

except FileNotFoundError:
    logging.error(f"Error: Tuner directory or project not found at 'keras_tuner_dir/{TUNER_PROJECT_NAME}'.")
    logging.error("Please ensure the directory exists and the project name is correct.")
except Exception as e:
    logging.error(f"Error loading tuner or extracting hyperparameters: {e}")
    logging.error("Please ensure 'keras_tuner_dir' and project name are correct, and N_FEATURES is set accurately.")