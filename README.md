# lasagne_CNN_framework

## main components:
1. train.py: it trains different networks specified by config.py (where the real network models are in models.py)
2. predict.py: Given a list of trained model, it loades all the models and make predictions
3. SETTINGS.py specifis the data directories
4. CNN_models: the specific CNN models
5. img_process: realtime loading and img augmentation (including rotation, shift, shear, flip, PCA_color noise, croping and more)
