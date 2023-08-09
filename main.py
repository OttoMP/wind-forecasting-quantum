import sys
import os
import numpy as np
from scipy import stats  
import tensorflow as tf
import pennylane as qml
import pandas as pd
from matplotlib import pyplot as plt

from quantum_neural_network import qnode_entangling
from statistics_1 import quantitative_analysis
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_history(history, n_layers):
    plt.figure(figsize=(14,5), dpi=320, facecolor='w', edgecolor='k')
    plt.title(f"Loss for depth {n_layers}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label="Loss/Epoch")
    plt.plot(history.history['val_loss'], label="Val Loss/Epoch")
    plt.legend()

    path = os.path.abspath(os.path.join(os.getcwd(), 'plots'))
    filename = f"experiment-{n_layers}.svg"
    plt.savefig(os.path.join(path,filename))


def carregar_tabela(path):
    dataframe=pd.read_csv(path)
    dataset = dataframe.values
    X_train_all=dataframe[['Total amount of precipitation','Vapor_Pressure','Relative_Humudity','Amount of cloudiness','Actual_Pressure']]
    y_train_all=dataframe[['Temperature']]

    return X_train_all, y_train_all.values


def main():

    ########################
    ### Importando dados ###
    ########################

    filename = sys.argv[1]

    X_all,y_all = carregar_tabela(filename)
    #y_train_all = y_train_all.reshape(-1,1)

    n_features = X_all.shape[1]
    n_instances = X_all.shape[0]
    print(f"There are {n_features} features and {n_instances} instâncias")
    print(X_all.head())
    print(y_all[:5])
    print("\n#########\n")

    ####################
    ### Scaling Data ###
    ####################
    
    scaler_x = StandardScaler()
    X_all_scaled = scaler_x.fit_transform(X_all)
    #scaler_y = StandardScaler()
    #y_all_scaled = scaler_y.fit_transform(y_all)

    #####################################
    ### Splitting Train and Test sets ###
    #####################################
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    # train is now 75% of the entire data set
    X_train, X_test, y_train, y_test = train_test_split(X_all_scaled, y_all, test_size=1 - train_ratio)
    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    
    #X_train = tf.cast(X_train, dtype=tf.float64)
    #y_train = tf.cast(y_train, dtype=tf.float64)
    print("Len(Train):",len(X_train))
    print("Len(Val):"  ,len(X_val))
    print("Len(Test):" ,len(X_test))

    print("\n#########\n")

    n_qubits = n_features
    print(f"Serão necessários {n_qubits} qubits")
    y_test_pred = np.array([])
    for n_layers in range(1,3):
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
        
        es=EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
        re=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min', min_lr=0.00001)
        history_model = model.fit(X_train, y_train
                                , epochs=30, batch_size=64
                                , callbacks=[es, re]
                                , verbose=1
                                , validation_data=(X_val, y_val))

        #################
        ### Loss Plot ###
        #################
        plot_history(history_model, n_layers)
        y_test_pred.append(model.predict(X_test,verbose=1))
        #y_test_pred_normal = scaler_y.inverse_transform(y_test_pred)
        print("\n#########\n")

    #####################
    ### Data Analysis ###
    #####################
    print("Len y_pred", len(y_test_pred))
    print(y_test_pred)
    erros_pd = quantitative_analysis(y_test, y_test_pred[0])
    print(erros_pd)
    print("\n#########\n")

    #mean_predictions, mean_error_normal, mean_error_left_normal, mean_error_right_normal = get_mean_left_right_error_interval(
    #model, scaler_x, X_val, y_val, y_test, y_test_pred)


if __name__ == "__main__":
    y_test = np.array(
        [[18.768929]
       , [14.344078]
       , [10.681574]
       , [17.663204]
       , [17.011688]
       , [14.130969]]) 
    y_test_pred = [np.array(
        [[22.581585 ]
       , [13.519949 ]
       , [7.6178145]
       , [19.135805 ]
       , [19.92104  ]
       , [12.70121  ]])]
    second_pred = np.array(
        [[22.581585 ]
       , [13.519949 ]
       , [7.6178145]
       , [19.135805 ]
       , [19.92104  ]
       , [12.70121  ]])+1

    print(y_test[:,0])
    print(len(y_test))

    print("---")
    print(y_test_pred)
    print(second_pred)
    #print(y_test_pred[0][:,0])
    y_test_pred.append(second_pred)
    print(y_test_pred)
    print(y_test_pred[0])
    print(y_test_pred[0][:,0])
    
    erros_pd = quantitative_analysis(y_test, y_test_pred)

    print(erros_pd)
    #main()