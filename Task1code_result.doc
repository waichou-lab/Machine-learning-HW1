TASK 1: FASHION MNIST CLASSIFICATION
============================================================

1. Loading Fashion MNIST dataset...
X_train_full shape: (60000, 28, 28)
X_train_full dtype: uint8
First training instance label: Coat

2. Building the neural network model...
C:\Users\www14\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\layers\reshaping\flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
2025-10-14 21:18:44.137320: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model summary:
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ flatten (Flatten)                    │ (None, 784)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 300)                 │         235,500 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 100)                 │          30,100 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 10)                  │           1,010 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 266,610 (1.02 MB)
 Trainable params: 266,610 (1.02 MB)
 Non-trainable params: 0 (0.00 B)

3. Examining model layers...
First hidden layer name: dense
Is get_layer('dense') same as hidden1? True
Weights shape: (784, 300)
Biases shape: (300,)
Biases initial values (first 10): [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

4. Compiling the model...

5. Training the model...
Epoch 1/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.6898 - loss: 0.9808 - val_accuracy: 0.8282 - val_loss: 0.5063
Epoch 2/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8241 - loss: 0.4989 - val_accuracy: 0.8392 - val_loss: 0.4633
Epoch 3/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8446 - loss: 0.4445 - val_accuracy: 0.8654 - val_loss: 0.4120
Epoch 4/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8520 - loss: 0.4195 - val_accuracy: 0.8658 - val_loss: 0.3906
Epoch 5/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.8608 - loss: 0.3896 - val_accuracy: 0.8692 - val_loss: 0.3791
Epoch 6/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8677 - loss: 0.3780 - val_accuracy: 0.8568 - val_loss: 0.3941
Epoch 7/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8700 - loss: 0.3679 - val_accuracy: 0.8650 - val_loss: 0.3741
Epoch 8/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8759 - loss: 0.3494 - val_accuracy: 0.8700 - val_loss: 0.3624
Epoch 9/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8786 - loss: 0.3428 - val_accuracy: 0.8802 - val_loss: 0.3445
Epoch 10/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8812 - loss: 0.3341 - val_accuracy: 0.8778 - val_loss: 0.3388
Epoch 11/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8844 - loss: 0.3253 - val_accuracy: 0.8770 - val_loss: 0.3441
Epoch 12/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8867 - loss: 0.3162 - val_accuracy: 0.8812 - val_loss: 0.3323
Epoch 13/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.8898 - loss: 0.3055 - val_accuracy: 0.8874 - val_loss: 0.3238
Epoch 14/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8902 - loss: 0.3041 - val_accuracy: 0.8838 - val_loss: 0.3238
Epoch 15/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8901 - loss: 0.3026 - val_accuracy: 0.8842 - val_loss: 0.3236
Epoch 16/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8954 - loss: 0.2919 - val_accuracy: 0.8798 - val_loss: 0.3263
Epoch 17/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8994 - loss: 0.2798 - val_accuracy: 0.8812 - val_loss: 0.3298
Epoch 18/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8985 - loss: 0.2800 - val_accuracy: 0.8884 - val_loss: 0.3095
Epoch 19/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8993 - loss: 0.2797 - val_accuracy: 0.8842 - val_loss: 0.3247
Epoch 20/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9017 - loss: 0.2702 - val_accuracy: 0.8930 - val_loss: 0.3018
Epoch 21/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9056 - loss: 0.2625 - val_accuracy: 0.8882 - val_loss: 0.3130
Epoch 22/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9067 - loss: 0.2621 - val_accuracy: 0.8880 - val_loss: 0.3113
Epoch 23/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9072 - loss: 0.2552 - val_accuracy: 0.8902 - val_loss: 0.2969
Epoch 24/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9098 - loss: 0.2485 - val_accuracy: 0.8888 - val_loss: 0.3119
Epoch 25/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9097 - loss: 0.2471 - val_accuracy: 0.8898 - val_loss: 0.3068
Epoch 26/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9133 - loss: 0.2407 - val_accuracy: 0.8890 - val_loss: 0.3019
Epoch 27/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9127 - loss: 0.2414 - val_accuracy: 0.8940 - val_loss: 0.2923
Epoch 28/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9166 - loss: 0.2323 - val_accuracy: 0.8848 - val_loss: 0.3230
Epoch 29/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9162 - loss: 0.2308 - val_accuracy: 0.8830 - val_loss: 0.3213
Epoch 30/30
1719/1719 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9211 - loss: 0.2239 - val_accuracy: 0.8978 - val_loss: 0.2839

6. Plotting learning curves...

7. Evaluating on test set...
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8530 - loss: 57.8253   
Test loss: 60.5007
Test accuracy: 0.8542

8. Making predictions on new instances...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step
Predicted probabilities (first instance):
  T-shirt/top: 0.00%
  Trouser: 0.00%
  Pullover: 0.00%
  Dress: 0.00%
  Coat: 0.00%
  Sandal: 0.00%
  Shirt: 0.00%
  Sneaker: 0.00%
  Bag: 0.00%
  Ankle boot: 100.00%
Predicted classes: ['Ankle boot', 'Pullover', 'Trouser']
Actual classes: ['Ankle boot', 'Pullover', 'Trouser']
