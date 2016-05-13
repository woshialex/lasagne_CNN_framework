# lasagne_CNN_framework

## main components:
1. load_data.py: It does realtime loading and img augmentation (including rotation, shift, shear, flip, PCA_color noise, croping and more)
2. train.py: it trains different networks specified by config.py (where the real network models are in models.py)
3. predict.py: Given a list of trained model, it loades all the models and make predictions
