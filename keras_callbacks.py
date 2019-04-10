from keras import backend as K
from keras.callbacks import Callback

class WRScheduler(Callback):
    """Warm restart scheduler for optimizers with decoupled weight decay.
    
    Warm restarts include cosine annealing with periodic restarts
    for both learning rate and weight decay. Normalized weight decay is also included.
    
    # Arguments
        steps_per_epoch: int > 0. The number of training batches per epoch.
        eta_min: float >=0. The minimum of the multiplier.
        eta_max: float >=0. The maximum of the multiplier.
        eta_decay: float >=0. The decay rate of eta_min/eta_max after each restart.
        cycle_length: int > 0. The number of epochs in the first restart cycle.
        cycle_mult_factor: float > 0. The rate to increase the number of epochs 
            in a cycle after each restart.
            
    # Reference
        - [SGDR: Stochastic Gradient Descent with Warm Restarts](http://arxiv.org/abs/1608.03983)
        - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    """
    
    def __init__(self,
                steps_per_epoch,
                eta_min=0.0,
                eta_max=1.0,
                eta_decay=1.0,
                cycle_length=10,
                cycle_mult_factor=1.5):

        super(WRScheduler, self).__init__()

        self.steps_per_epoch = steps_per_epoch

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_decay = eta_decay

        self.steps_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.cycle_mult_factor = cycle_mult_factor

    def cal_eta(self):
        '''Calculate eta'''
        fraction_to_restart = self.steps_since_restart / (self.steps_per_epoch * self.cycle_length)
        eta = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + np.cos(fraction_to_restart * np.pi))
        return eta
    
    def on_train_begin(self, logs={}):
        '''Set the number of training batches of the first restart cycle to steps_per_cycle'''
        K.set_value(self.model.optimizer.steps_per_cycle, self.steps_per_epoch * self.cycle_length)

    def on_train_batch_begin(self, batch, logs={}):
        '''update eta'''
        eta = self.cal_eta()
        K.set_value(self.model.optimizer.eta, eta)
        self.steps_since_restart += 1

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary'''
        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.cycle_mult_factor)
            self.next_restart += self.cycle_length
            self.eta_min *= self.eta_decay
            self.eta_max *= self.eta_decay
            K.set_value(self.model.optimizer.steps_per_cycle, self.steps_per_epoch * self.cycle_length)
