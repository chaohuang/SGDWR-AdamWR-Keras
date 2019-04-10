# SGDW/SGDWR and AdamW/AdamWR optimizers for Keras
Keras implementations of SGDW and AdamW (SGD and Adam with decoupled weight decay), which can be used with warm restarts to obtain SGDWR and AdamWR.

## Usage
```
from keras_optimizers import SGDW

optimizer = SGDW(lr=0.01, weight_decay=0.01)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
              
model.fit(x_train, y_train)
```
For SGDWR/AdamWR, use the callback `WRScheduler` with SGDW/AdamW
```
from keras_optimizers import AdamW
from keras_callbacks import WRScheduler

optimizer = AdamW(lr=0.001, weight_decay=0.01)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
              
cb_wr = WRScheduler(steps_per_epoch=len(x_train)/batch_size)

model.fit(x_train, y_train, callbacks=[cb_wr])
```

## Tested on this system
- Python 3.6.8
- TensorFlow 1.12.0
- Keras 2.2.4

## Reference
[SGDR: Stochastic Gradient Descent with Warm Restarts](http://arxiv.org/abs/1608.03983), Ilya Loshchilov, Frank Hutter

[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), Ilya Loshchilov, Frank Hutter
