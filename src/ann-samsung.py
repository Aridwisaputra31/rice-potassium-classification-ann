Lampiran 17. Kode Program Jaringan Saraf Tiruan Kamera Samsung 
import os 
import pandas as pd 
import numpy as np 
import time 
import seaborn as sns 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.metrics import confusion_matrix, accuracy_score 
MK = 'samsung_final3' # merk kamera 
print('PEMODELAN NEURAL NETWORK UNTUK CITRA DARI TELPON ' + MK.upper()) 
# BACA DATA 
t0 = time.perf_counter() 
print('\nBaca data file:') 
nama_file = 'Train_JST_Samsung_3.xlsx' 
nama_folder = r'D:\Penelitian\Program penelitian\Program 
Jadi\JST\Bagi_data\Samsung\702010' 
nf = os.path.join(nama_folder, nama_file) 
print(nf) 
# Train data 
df = pd.read_excel(nf, sheet_name='Sheet1') 
x_train = df[df.columns[1:6]]  # input 
y_train = df[df.columns[6:7]]  # target 
print('\nData Training') 
print('\nX: ', x_train) 
print('\nY0: ', y_train) 
nama_file = 'Val_JST_Samsung_3.xlsx' 
nama_folder = r'D:\Penelitian\Program penelitian\Program 
Jadi\JST\Bagi_data\Samsung\702010' 
nf = os.path.join(nama_folder, nama_file) 
# Valid data 
df2 = pd.read_excel(nf, sheet_name='Sheet1') 
x_valid = df2[df2.columns[1:6]]  # input 
y_valid = df2[df2.columns[6:7]]  # target 
print('\nData Validation') 
print('\nX: ', x_valid) 
print('\nY0: ', y_valid) 
# NORMALISASI DATA 
sc = StandardScaler() 
x_train_std = sc.fit_transform(x_train) 
x_valid_std = sc.fit_transform(x_valid) 
# One-hot encoding untuk target 
ohe = OneHotEncoder() 

y_valid_trans = ohe.fit_transform(y_valid).toarray() 
# MEMBUAT MODEL JST 
f_act = 'relu' 
tf.config.set_visible_devices([], 'GPU') 
model = tf.keras.Sequential() 
model.add(tf.keras.layers.Dense(350, input_dim=5, activation=f_act)) #, 
kernel_regularizer=l2 
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.Dense(360, activation=f_act)) 
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.Dense(370, activation=f_act)) 
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.Dense(380, activation=f_act)) 
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.Dense(400, activation=f_act)) 
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.Dense(3, activation='softmax')) 
lr=0.00003 
opt = tf.keras.optimizers.Adam(learning_rate=lr)  # initial learning rate 
model.compile(loss='categorical_crossentropy',  
optimizer=opt,  
metrics=['accuracy']) 
model.summary() 
# TRAINING 
jml_epoch = 12000 
print('Training...') 
history = model.fit(x_train_std,  
y_train_trans,  
validation_data=(x_valid_std, y_valid_trans),  
epochs=jml_epoch,  
batch_size=100) 
# Confusion matrix dan accuracy untuk data training 
y_pred_train = model.predict(x_train_std) 
y_train_class = np.argmax(y_train_trans, axis=1) 
y_pred_train_class = np.argmax(y_pred_train, axis=1) 
cm_train = confusion_matrix(y_train_class, y_pred_train_class) 
acc_train = accuracy_score(y_train_class, y_pred_train_class) 
print('\nConfusion Matrix Samsung - Training:') 
print(cm_train) 
print('Training Accuracy= ', acc_train) 
label_cm=["Cukup", "Kurang", "Lebih"] 
sns.heatmap(cm_train, annot=True, fmt="d", cmap="viridis", 
xticklabels=label_cm, yticklabels=label_cm) 
plt.title("Confusion Matrix Samsung - Training") 
plt.xlabel("Predicted Class") 
plt.ylabel("True Class")
y_train_trans = ohe.fit_transform(y_train).toarray()
color="black", ha="center") 
plt.show() 
# Visualisasi Training 
plt.figure() 
plt.plot(history.history['accuracy'], label='Train Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.title('Model Accuracy Samsung') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show() 
plt.figure() 
plt.plot(history.history['loss'], label='Train Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.title('Model Loss Samsung') 
plt.xlabel('Epoch') 
plt.ylabel('Loss') 
plt.legend() 
plt.show() 
# Simpan model hasil training 
nf_ann_json = 'model_' + MK + '.json' 
nf_weights_h5 = 'model_' + MK + '.weights.h5' 
model_json = model.to_json() 
with open(nf_ann_json, 'w') as json_file: 
json_file.write(model_json) 
model.save_weights(nf_weights_h5) 
print('Model hasil training telah disimpan dalam:') 
print('Model ANN --->', nf_ann_json) 
print('Bobot ------->', nf_weights_h5)
