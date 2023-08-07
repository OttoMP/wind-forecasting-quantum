# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import tensorflow as tf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers, activations
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

from math import sqrt
from scipy import stats
from datetime import datetime,timedelta

seed = 7
np.random.seed(seed)

from tensorflow.keras import backend
backend.set_floatx('float64')

# %% [markdown]
# ## Carregando os Dados

# %%
def carregar_tabela(path):
    dataframe=pd.read_csv(path)
    dataset = dataframe.values
    X_train_all=dataframe[['Total amount of precipitation','Vapor_Pressure','Relative_Humudity','Amount of cloudiness','Actual_Pressure']]
    y_train_all=dataframe[['Temperature']]

    return X_train_all, y_train_all.values

# %%
filename = '2023Ogur/WeatherForcasting_DATA.csv'

X_train_all,y_train_all = carregar_tabela(filename)
#y_train_all = y_train_all.reshape(-1,1)

n_features = X_train_all.shape[1]
n_instances = X_train_all.shape[0]
print(f"There are {n_features} features and {n_instances} instâncias")
X_train_all.head()

# %%
y_train_all

# %% [markdown]
# ## Normalizando

# %%
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train_all)
#scaler_y = StandardScaler()
#y_train_scaled = scaler_y.fit_transform(y_train_all)

# %%
y_train_all

# %%
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train_all, test_size=0.9, random_state=42)
X_train = tf.cast(X_train, dtype=tf.float64)
y_train = tf.cast(y_train, dtype=tf.float64)
print("Len(Train):",len(X_train))
print("Len(Test):",len(X_val))

# %% [markdown]
# ## Rede Quântica

# %%
def H_layer(n_qubits):
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

def Data_AngleEmbedding_layer(inputs, n_qubits):
    qml.templates.AngleEmbedding(inputs,rotation='Y', wires=range(n_qubits))

def RY_layer(w):
    print(w.shape)
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def ROT_layer(w):
    for i in range(5):
        qml.Rot(*w[i],wires=i)

def strong_entangling_layer(nqubits):
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[3,4])
    qml.CNOT(wires=[4,0])
    
    
def entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2): 
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  
        qml.CNOT(wires=[i, i + 1])

# %%
n_qubits = n_features
print(f"Serão necessários {n_qubits} qubits")
n_layers = 1

#dev = qml.device('lightning.gpu', wires=n_qubits)
#dev = qml.device('lightning.qubit', wires=n_qubits)
dev = qml.device('default.qubit', wires=n_qubits)
@qml.qnode(dev, interface="torch")
def qnode(inputs, weights_1):
    H_layer(n_qubits)
    Data_AngleEmbedding_layer(inputs, n_qubits)
    for k in range(n_layers):
        entangling_layer(n_qubits)
        ROT_layer(weights_1[k])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# %% [markdown]
# ## Drawing Quantum Circuit

# %%
weight_shapes = {"weights_1": (n_layers,n_qubits,3)}

sampl_weights = np.random.uniform(low=0, high=np.pi, size=weight_shapes["weights_1"])
sampl_input = np.random.uniform(low=0, high=np.pi, size=(n_qubits,))
print(qml.draw(qnode, expansion_strategy="device")(sampl_input, sampl_weights))

# %%
q_layer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
Activation=tf.keras.layers.Activation(activations.linear)
output_layer = tf.keras.layers.Dense(1,kernel_initializer='normal')

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

model = tf.keras.models.Sequential([q_layer,Activation, output_layer])
model.compile(opt, loss="mse")

# %%
input_shape = (n_qubits,)

model.build(input_shape)
model.summary()

# %%
es=EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
re=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min', min_lr=0.00001)
history_model = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=[re], verbose=1, validation_data=(X_val, y_val))

# %% [markdown]
# ## Criando gráfico de Loss por Epoch

# %%
def plot_history(history):
    plt.figure(figsize=(14,5), dpi=320, facecolor='w', edgecolor='k')
    plt.title("Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label="Loss/Epoch")
    plt.plot(history.history['val_loss'], label="Val Loss/Epoch")
    plt.legend()

    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'plots'))
    filename = "experiment-1.png"
    plt.savefig(os.path.join(path,filename))

# %%
plot_history(history_model)

# %% [markdown]
# ## Validação

# %%
y_predict=model.predict(X_val,verbose=1)
#predict_normal = scaler_y.inverse_transform(predict)
y_predict

# %%
print('Mean Squared Error:', mean_squared_error(y_val, y_predict))

# %%
def factor_of_2(y_true, y_pred):
    min_ = 0.5
    max_ = 2.0

    tensor_true = tf.constant(y_true)
    tensor_true = tf.cast(tensor_true, tf.float32)
    tensor_pred = tf.constant(y_pred)
    tensor_pred = tf.cast(tensor_pred, tf.float32)

    division = tf.divide(tensor_pred, tensor_true)

    greater_min = tf.greater_equal(division, min_)
    less_max = tf.less_equal(division, max_)

    res = tf.equal(greater_min, less_max)
    res = tf.cast(res, tf.float32)

    return backend.get_value(tf.reduce_mean(res))


def allmetrics(original,predito):
    r_value = 0
    slope, intercept, r_value, p_value, std_err = stats.linregress(original, predito)
    mse = mean_squared_error(original, predito)
    mae = mean_absolute_error(original, predito)
    rr = r2_score(original,predito)
    pea = stats.pearsonr(original, predito)
    fat = factor_of_2(original,predito)
    nmse = mse/stats.tvar(original)
    rmse = sqrt(mse)
    nrmse = rmse/stats.tstd(original)
    return mae,mse,nmse,r_value,rr,fat,rmse,nrmse

# %%
def get_error_interval(model, X_val, Y_val, y_test_pred, p_value):
    y_val_pred = model.predict(X_val)
    y_val_error = np.abs(Y_val - y_val_pred)
    error_quantile=np.ndarray((1,Y_val.shape[1]));
    for i in range(Y_val.shape[1]):
        error_quantile[0,i] = np.quantile(y_val_error[:,i], q=p_value, interpolation='higher')
        
    y_test_interval_pred_left=np.ndarray(y_test_pred.shape);
    y_test_interval_pred_right=np.ndarray(y_test_pred.shape);
    
    for i in range(y_test_pred.shape[1]):
        y_test_interval_pred_left[:,i] = y_test_pred[:,i] - error_quantile[0,i]
        y_test_interval_pred_right[:,i] = y_test_pred[:,i] + error_quantile[0,i]
    return error_quantile, y_test_interval_pred_left, y_test_interval_pred_right

# %%
def get_mean_left_right_error_interval(model, y_test, y_scaler, y_test_pred):
    error, error_left, error_right = get_error_interval(model, X_val, y_val, y_test_pred, 0.95)
    
    #error_left_normal = y_scaler.inverse_transform(error_left)
    #error_right_normal = y_scaler.inverse_transform(error_right)

    error_left_normal = error_left
    error_right_normal = error_right

    mean_error_normal=np.ndarray((1,y_test.shape[1]));
    mean_error_left_normal=np.ndarray((1,y_test.shape[1]));
    mean_error_right_normal=np.ndarray((1,y_test.shape[1]));
    mean_predictions=np.ndarray((1,y_test_pred.shape[1]));

    for i in range(y_test.shape[1]):
        mean_error_left_normal[0,i] = np.mean(error_left_normal[:,i])
        mean_error_right_normal[0,i] = np.mean(error_right_normal[:,i])
        mean_predictions[0,i]=np.mean(y_test_pred[:,i])

    mean_error_normal=(mean_error_right_normal-mean_error_left_normal)/2
    return mean_predictions, mean_error_normal, mean_error_left_normal, mean_error_right_normal

# %%
mean_predictions, mean_error_normal, mean_error_left_normal, mean_error_right_normal = get_mean_left_right_error_interval(
    model, y_val, scaler_x, y_predict)

# %%
def get_plot_prediction_versus_observed(model, y_test, predict):
    valores = []
    for i in range(y_test.shape[1]):
        mae,mse,nmse,r_value,rr,fat,rmse,nrmse = allmetrics(y_test[:,i],predict[:,i])
        valores.append([str(i+1)+" depth",mae,mse,nmse,rmse,nrmse,r_value,rr,fat,mean_error_normal[0,i],mean_error_left_normal[0,i],mean_predictions[0,i],mean_error_right_normal[0,i]])
        print("MAE:",mae)
        print("MSE:",mse)
        print("NMSE:",nmse)
        print("RMSE:",rmse)
        print("NRMSE:",nrmse)
        print("R:",r_value)
        print("R²:",rr)
        print("Fator de 2:",fat)

    erros = pd.DataFrame(valores)
    erros.columns = ['Index','MAE','MSE','NMSE','RMSE','NRMSE','R','R²','Fator de 2', 'error interval (+/-)', 'left limit', 'mean', 'right limit']
    erros = erros.set_index('Index')
    erros.loc['Média'] = erros.mean()
    return erros


# %%
erros_pd=get_plot_prediction_versus_observed(model, y_val, y_predict)
erros_pd


