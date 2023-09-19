import sys
import os
import numpy as np
import tensorflow as tf
import pennylane as qml
import pandas as pd
from matplotlib import pyplot as plt

from quantum_neural_network import qnode_entangling, qnode_strong_entangling
from statistics_1 import quantitative_analysis, get_mean_left_right_error_interval
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def plot_history(history, n_layers):
    plt.figure(figsize=(14,5), dpi=320, facecolor='w', edgecolor='k')
    plt.title(f"Loss for depth {n_layers}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label="Loss/Epoch")
    plt.plot(history.history['val_loss'], label="Val Loss/Epoch")
    plt.xticks(range(0, len(history.history['loss'])+1, 5))
    plt.legend()
    plt.grid()

    values = np.array([list(range(1, len(history.history['loss'])+1)), history.history['loss'], history.history['val_loss']])
    loss_pd = pd.DataFrame(np.transpose(values))
    loss_pd.columns = ["Época", "Loss", "Val Loss"]
    loss_pd = loss_pd.set_index("Época")
    path = os.path.abspath(os.path.join(os.getcwd(), 'analysis'))
    filename = f"loss-df-{(n_layers)}.csv"
    loss_pd.to_csv(os.path.join(path,filename))

    path = os.path.abspath(os.path.join(os.getcwd(), 'plots'))
    filename = f"loss-history-{n_layers}.svg"
    plt.savefig(os.path.join(path,filename))


def plot_prediction_versus_observed(n_layers, y_test, y_pred, mean_error_normal):
    for i in range(y_test.shape[1]):
        plt.figure(figsize=(20,5), dpi=320, facecolor='w', edgecolor='k')
        plt.title("Previsão do vento para "+str(i+1)+" hora(s) à frente com "+str(n_layers)+" camadas")
        plt.xlabel("Amostras")
        plt.ylabel("Velocidade do Vento (m/s)")
        plt.plot(y_pred[:,i], label="Predito", color='blue')
        plt.fill_between(range(y_pred.shape[0]), y_pred[:,i]-mean_error_normal[0,i], y_pred[:,i]+mean_error_normal[0,i], color='blue', alpha=0.05)
        plt.plot(y_test[:,i], label="Original", color='orange')
        plt.legend()
        path = os.path.abspath(os.path.join(os.getcwd(), 'plots'))
        filename = f"prediction-{n_layers}.svg"
        plt.savefig(os.path.join(path,filename))


def carregar_tabela(path):
    X_train_all=pd.read_csv(path, sep='\t', header = 0)
    y_train_all = X_train_all[:].drop(X_train_all.index[0])
    X_train_all = X_train_all.iloc[:-3,:]
    y_train_all['1h - Vento'] = y_train_all.iloc[:,4].shift(0)
    y_train_all= y_train_all.iloc[:-2, -1:]

    return X_train_all, y_train_all.values


def main():

    ########################
    ### Importando dados ###
    ########################

    filename = 'data/train150_mucuri.txt'

    X_all,y_all = carregar_tabela(filename)

    n_features = X_all.shape[1]
    n_instances = X_all.shape[0]
    print(f"There are {n_features} features and {n_instances} instâncias")
    print(X_all.head())
    print(y_all[:5])
    print("\n#########\n")

    ####################
    ### Scaling Data ###
    ####################
    
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    X_all_scaled = scaler_x.fit_transform(X_all)
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_all_scaled = scaler_y.fit_transform(y_all)

    #####################################
    ### Splitting Train and Test sets ###
    #####################################
    train_ratio = 0.8
    X_train, X_val, y_train, y_val = train_test_split(X_all_scaled, y_all_scaled, test_size=1 - train_ratio)

    test = 'data/prev150_mucuri.txt'
    X_test,y_test = carregar_tabela(test)
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    #X_train = tf.cast(X_train, dtype=tf.float64)
    #y_train = tf.cast(y_train, dtype=tf.float64)
    print("Len(Train):",len(X_train))
    print("Len(Val):"  ,len(X_val))
    print("Len(Test):" ,len(X_test_scaled))

    print("\n#########\n")

    n_qubits = n_features
    print(f"Serão necessários {n_qubits} qubits")
    list_y_pred = []
    for n_layers in range(1,9):
        ##########################################
        ### Creating Neural Network with Keras ###
        ##########################################
        print(f"Training with depth {n_layers}")
        weight_shapes = {"weights": (n_layers,n_qubits,3)}

        q_layer = qml.qnn.KerasLayer(qnode_entangling, weight_shapes, output_dim=n_qubits)
        Activation=tf.keras.layers.Activation(tf.keras.activations.linear)
        output_layer = tf.keras.layers.Dense(1,kernel_initializer='normal')

        opt = tf.keras.optimizers.Adam(learning_rate=0.1)

        model = tf.keras.models.Sequential([q_layer,Activation, output_layer])
        model.compile(opt, loss="mse")

        input_shape = (n_qubits,)

        model.build(input_shape)
        print(model.summary())

        ######################
        ### Training Model ###
        ######################
        
        es=EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        re=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='min', min_lr=0.00001)
        history_model = model.fit(X_train, y_train
                                , epochs=50, batch_size=32
                                , callbacks=[re]
                                , verbose=0
                                , validation_data=(X_val, y_val))

        #################
        ### Loss Plot ###
        #################
        plot_history(history_model, n_layers)

        ##################
        ### Prediction ###
        ##################
        y_pred = model.predict(X_test_scaled,verbose=0)
        y_pred_normal = scaler_y.inverse_transform(y_pred)
        list_y_pred.append(y_pred_normal)
        mean_predictions, mean_error_normal, mean_error_left_normal, mean_error_right_normal = get_mean_left_right_error_interval(model, scaler_y, X_val, y_val, y_test, y_pred_normal)
        plot_prediction_versus_observed(n_layers, y_test, y_pred_normal, mean_error_normal)
        print("\n#########\n")


    #####################
    ### Data Analysis ###
    #####################
    print("Len list_y_pred", len(list_y_pred))
    #print(list_y_pred)
    all_analysis = quantitative_analysis(y_test, list_y_pred)
    print(all_analysis)
    print("\n#########\n")

    #error_interval = get_mean_left_right_error_interval(model, scaler_x, X_val, y_val, y_test, y_test_pred)
    #print(error_interval)
    #print("\n#########\n")

    #all_analysis = erros_pd.join(error_interval)
    #print(all_analysis)
    
    path = os.path.abspath(os.path.join(os.getcwd(), 'analysis'))
    filename = "metrics"+"-"+filename.split("/")[1]
    all_analysis.to_csv(os.path.join(path,filename))

if __name__ == "__main__":
    main()
