============================================================
TASK 2: CALIFORNIA HOUSING REGRESSION
============================================================

1. Loading California housing dataset...
Training set shape: (11610, 8)
Validation set shape: (3870, 8)
Test set shape: (5160, 8)

2. Building Sequential regression model...
C:\Users\www14\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\layers\core\dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Sequential model summary:
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense_3 (Dense)                      │ (None, 30)                  │             270 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (Dense)                      │ (None, 1)                   │              31 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 301 (1.18 KB)
 Trainable params: 301 (1.18 KB)
 Non-trainable params: 0 (0.00 B)

3. Training Sequential model...
Epoch 1/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.5944 - val_loss: 1.6136
Epoch 2/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4623 - val_loss: 1.2918
Epoch 3/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 992us/step - loss: 0.4245 - val_loss: 9.0748
Epoch 4/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.7000 - val_loss: 11.4305
Epoch 5/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.3870 - val_loss: 9.0549
Epoch 6/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 947us/step - loss: 0.7443 - val_loss: 0.3526
Epoch 7/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 995us/step - loss: 0.3760 - val_loss: 0.3534
Epoch 8/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 959us/step - loss: 0.3761 - val_loss: 0.3640
Epoch 9/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.3686 - val_loss: 0.6340
Epoch 10/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 957us/step - loss: 0.3632 - val_loss: 0.3350
Epoch 11/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.3602 - val_loss: 0.3315
Epoch 12/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.3433 - val_loss: 0.3316
Epoch 13/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 938us/step - loss: 0.3427 - val_loss: 0.3317
Epoch 14/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.3473 - val_loss: 0.3323
Epoch 15/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 998us/step - loss: 0.3522 - val_loss: 0.3273
Epoch 16/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.3476 - val_loss: 0.3253
Epoch 17/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.3423 - val_loss: 0.3220
Epoch 18/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 887us/step - loss: 0.3260 - val_loss: 0.3282
Epoch 19/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 823us/step - loss: 0.3513 - val_loss: 0.3244
Epoch 20/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 834us/step - loss: 0.3391 - val_loss: 0.3288

4. Evaluating Sequential model...
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 590us/step - loss: 0.3448
Test MSE: 0.3508
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
Sequential model predictions: [0.7633079 1.6731818 4.2489753]
Actual values: [0.477   0.458   5.00001]

5. Building Wide & Deep model using Functional API...
Wide & Deep model summary:
Model: "functional_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer_2 (InputLayer)    │ (None, 8)                 │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_5 (Dense)               │ (None, 30)                │             270 │ input_layer_2[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_6 (Dense)               │ (None, 30)                │             930 │ dense_5[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ concatenate (Concatenate)     │ (None, 38)                │               0 │ input_layer_2[0][0],       │
│                               │                           │                 │ dense_6[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_7 (Dense)               │ (None, 1)                 │              39 │ concatenate[0][0]          │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 1,239 (4.84 KB)
 Trainable params: 1,239 (4.84 KB)
 Non-trainable params: 0 (0.00 B)

6. Training Wide & Deep model...
Epoch 1/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 2.0225 - val_loss: 1.0437
Epoch 2/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 883us/step - loss: 0.7749 - val_loss: 0.7295
Epoch 3/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 872us/step - loss: 0.6997 - val_loss: 0.6331
Epoch 4/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 847us/step - loss: 0.6260 - val_loss: 0.6087
Epoch 5/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 863us/step - loss: 0.6062 - val_loss: 0.7031
Epoch 6/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 831us/step - loss: 0.5615 - val_loss: 0.5290
Epoch 7/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 854us/step - loss: 0.5647 - val_loss: 0.4978
Epoch 8/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 864us/step - loss: 0.5095 - val_loss: 0.4960
Epoch 9/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 832us/step - loss: 0.4962 - val_loss: 0.4596
Epoch 10/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 861us/step - loss: 0.4871 - val_loss: 0.5115
Epoch 11/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 857us/step - loss: 0.4692 - val_loss: 0.4452
Epoch 12/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 851us/step - loss: 0.4546 - val_loss: 0.4979
Epoch 13/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 867us/step - loss: 0.4613 - val_loss: 0.4181
Epoch 14/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 877us/step - loss: 0.4533 - val_loss: 0.4248
Epoch 15/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 857us/step - loss: 0.4432 - val_loss: 0.4826
Epoch 16/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 851us/step - loss: 0.4106 - val_loss: 0.4063
Epoch 17/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 856us/step - loss: 0.4216 - val_loss: 0.4432
Epoch 18/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 852us/step - loss: 0.4123 - val_loss: 0.4140
Epoch 19/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 914us/step - loss: 0.3975 - val_loss: 0.4126
Epoch 20/20
363/363 ━━━━━━━━━━━━━━━━━━━━ 0s 856us/step - loss: 0.4139 - val_loss: 0.4717

7. Evaluating Wide & Deep model...
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 775us/step - loss: 0.4074
Wide & Deep Test MSE: 0.4063

8. Model Comparison:
Sequential Model Test MSE: 0.3508
Wide & Deep Model Test MSE: 0.4063
