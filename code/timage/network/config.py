
def configure_seed(seed=42):
    """"
        See explanations
        https: // machinelearningmastery.com / reproducible - results - neural - networks - keras /
        https://keras.io/getting-started/faq/
    """
    seed = int(seed)
    import numpy as np
    import tensorflow as tf
    import random as rn
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)
