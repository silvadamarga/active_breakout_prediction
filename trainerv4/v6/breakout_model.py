#!/usr/bin/env python3
"""
v6 - Breakout Model Architecture
Defines the Keras model (CNN + GRU/LSTM) for 3-class classification.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, optimizers, backend as K
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, LSTM, GRU, MaxPooling1D, Bidirectional,
    GlobalAveragePooling1D, LayerNormalization, Masking, Multiply, Softmax
)

# Import config
from .config import TIME_STEPS

def build_model_from_hp(hp, n_features):
    """Builds model for Keras Tuner, MODIFIED for 3-class output."""
    conv_filters_1 = hp.Int('conv_filters_1', 32, 128, step=32, default=64)
    conv_filters_2 = hp.Int('conv_filters_2', 32, 64, step=16, default=48)
    kernel_size = hp.Choice('kernel_size', [3, 5], default=5)
    rnn_type = hp.Choice('rnn_type', ['gru', 'lstm'], default='gru')
    rnn_units = hp.Int('rnn_units', 32, 128, step=32, default=64)
    shared_units = hp.Int('shared_units', 32, 96, step=32, default=64)
    head_units = hp.Int('head_units', 16, 64, step=16, default=32)
    dropout = hp.Float('dropout', 0.1, 0.4, step=0.05, default=0.2)
    lr = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4], default=1e-3)
    use_attention = hp.Boolean('use_attention', default=True)

    inp = Input(shape=(TIME_STEPS, n_features), name='inp')
    x = Conv1D(filters=conv_filters_1, kernel_size=kernel_size, activation='relu', padding='same')(inp)
    x = Conv1D(filters=conv_filters_2, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LayerNormalization()(x)

    if rnn_type == 'lstm':
        rnn_layer = LSTM(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=0.2)
    else:
        rnn_layer = GRU(units=rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=0.2, reset_after=False)

    x = Bidirectional(rnn_layer)(x)
    x = LayerNormalization()(x)

    if use_attention:
        score = Dense(1, activation='tanh')(x)
        score = Softmax(axis=1)(score)
        x = Multiply()([x, score])
        x = GlobalAveragePooling1D()(x)
    else:
        x = GlobalAveragePooling1D()(x)

    x = Dropout(dropout)(x)
    shared = Dense(shared_units, activation='relu')(x)
    shared = LayerNormalization()(shared)
    
    head = Dense(head_units, activation='relu')(shared)
    head = Dropout(dropout)(head)
    
    # --- NEW MODEL HEAD ---
    # Single 3-class output (0:Loss, 1:Timeout, 2:Win)
    output = Dense(3, activation='softmax', name='output', dtype='float32')(head)
    # --- END NEW HEAD ---

    model = Model(inputs=inp, outputs=output)
    
    # --- NEW COMPILE STEP ---
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=lr, weight_decay=1e-5, clipnorm=1.0),
        loss='categorical_crossentropy', # Changed from dict
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc_micro', curve='ROC', multi_label=False)]
    )
    # --- END NEW COMPILE STEP ---
    return model

class HPMock:
    """Mocks the Keras Tuner HP object to build a model from a dict."""
    def __init__(self, values): self.values = values
    def Int(self, name, *args, **kwargs): return int(self.values.get(name, kwargs.get('default')))
    def Float(self, name, *args, **kwargs): return float(self.values.get(name, kwargs.get('default')))
    def Choice(self, name, *args, **kwargs): return self.values.get(name, kwargs.get('default'))
    def Boolean(self, name, *args, **kwargs): return bool(self.values.get(name, kwargs.get('default')))

def build_model_from_dict(hp_values: dict, n_features: int):
    """Builds model from a dict, MODIFIED for 3-class output."""
    hp_mock = HPMock(hp_values)
    return build_model_from_hp(hp_mock, n_features)
