import gc
import os.path


class Checkpoints:

    def __init__(self, best_model, best_value):

        self.best_model = best_model
        self.best_value = best_value

    def get_best_model(self):
        return self.best_model

    def update(self, current_model, new_value):
        if new_value < self.best_value:
            print('*** Val loss IMPROVED from {:.4f} to {:.4f} ***'.format(self.best_value, new_value))
            self.best_value = new_value
            self.best_model = current_model
        else:
            print('Val loss of DID NOT improve from {:.4f}'.format(self.best_value))

        gc.collect()


