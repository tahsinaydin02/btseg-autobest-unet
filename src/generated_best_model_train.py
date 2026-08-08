# generated_best_model_train.py
# auto-generated at 2025-12-29 20:24:12

import train_variants as base

MODEL_SPECS = {
    'UNet-AutoBest': {'channels': (24, 32, 80), 'strides': (2, 2), 'num_res_units': 2},
}

if __name__ == '__main__':
    base.MODEL_SPECS = MODEL_SPECS
    base.train_all(max_epochs=100, val_fraction=0.2, batch_size=1)
