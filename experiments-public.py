import os
import tensorflow as tf
import pennylane as qml
import pandas as pd
from matplotlib import pyplot as plt
from numpy import array as np_array

from src.circuits import qnode_circular_entangling
from src.statistics import quantitative_analysis, get_mean_left_right_error_interval, verify_distribution_wilcoxtest
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

    path = os.path.abspath(os.path.join(os.getcwd(), 'plots'))
    filename = f"loss-history-public-{n_layers}.png"
    plt.savefig(os.path.join(path,filename))

    values = np_array([list(range(1, len(history.history['loss'])+1)), history.history['loss'], history.history['val_loss']])
    loss_pd = pd.DataFrame(np.transpose(values))
    loss_pd.columns = ["Epoch", "Loss", "Val Loss"]
    loss_pd = loss_pd.set_index("Epoch")
    path = os.path.abspath(os.path.join(os.getcwd(), 'analysis'))
    filename = f"loss-public-{n_layers}-layers.csv"
    loss_pd.to_csv(os.path.join(path,filename))


def plot_prediction_versus_observed(n_layers, y_test, y_pred, mean_error_normal):
    for i in range(y_test.shape[1]):
        plt.figure(figsize=(20,5), dpi=320, facecolor='w', edgecolor='k')
        plt.title(f"Temperature Forecast for {i+1} hours ahead for {n_layers} layers")
        plt.xlabel("Samples")
        plt.ylabel("Temperature (°C)")
        plt.plot(y_pred[:,i], label="Prediction", color='blue')
        plt.fill_between(range(y_pred.shape[0]), y_pred[:,i]-mean_error_normal[0,i], y_pred[:,i]+mean_error_normal[0,i], color='blue', alpha=0.05)
        plt.plot(y_test[:,i], label="Original", color='orange')
        plt.legend()
        path = os.path.abspath(os.path.join(os.getcwd(), 'plots'))
        filename = f"prediction-public-{n_layers}-layers-{i+1}-hours.png"
        plt.savefig(os.path.join(path,filename))


def carregar_tabela(path, prev):
    X=pd.read_csv(path)
    X.dropna(axis=0,how='any',inplace=True)
    
    # Using only the target column and removing the first index to predict the next target
    y = X[:].drop(X.index[0])
    
    # Remove the last line of X because the predicted Y will not have an extra line
    X = X.iloc[:-prev,:]
    
    for i in range(prev):
        y[f'Prev {i+1} step'] = y.iloc[:,-1].shift(-i)
    if prev-1 == 0:
        y = y.iloc[:, -prev:]
    else:
        y = y.iloc[:-prev+1, -prev:]

    return X, y.values


def main():

    ######################
    ### Importing Data ###
    ######################
    prev = 1
    path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    filename = "WeatherForecasting.csv"
    dataroot = os.path.join(path,filename)

    print(f"importing data from {dataroot}")
    X_all,y_all = carregar_tabela(dataroot, prev)
    
    # Splitting dataframe
    X_train_val = X_all.iloc[:int(X_all.shape[0]*.7),:]
    y_train_val = y_all[:int(X_all.shape[0]*.7)]
    X_test      = X_all.iloc[int(X_all.shape[0]*.7):,:]
    y_test      = y_all[int(X_all.shape[0]*.7):] 
    
    n_instances = X_train_val.shape[0]
    n_features  = X_train_val.shape[1]
    print(f"There are {n_features} features and {n_instances} instances in Train set")
    print(X_all.head())
    print(f"There are {X_test.shape[1]} features and {X_test.shape[0]} instances in Test set")
    print(X_test.head())
    print("Size y Train", len(y_train_val),"\n", y_all[:5])
    print("Size y Test", len(y_test),"\n", y_test[:5])
    print("\n#########\n")
    
    ####################
    ### Scaling Data ###
    ####################
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(X_train_val)
    X_train_val_scaled = scaler_x.transform(X_train_val)
    X_test_scaled = scaler_x.transform(X_test)

    #####################################
    ### Splitting Train and Test sets ###
    #####################################
    train_ratio = 0.8
    X_train, X_val, y_train, y_val = train_test_split(X_train_val_scaled, y_train_val, test_size=1 - train_ratio)

    #X_train = tf.cast(X_train, dtype=tf.float64)
    #y_train = tf.cast(y_train, dtype=tf.float64)
    print(f"Len(Train): {len(X_train)}")
    print(f"Len(Val): {len(X_val)}")
    print(f"Len(Test): {len(X_test_scaled)}")

    print("\n#########\n")
    
    n_qubits = n_features
    print(f"Circuit size: {n_qubits} qubits")
    list_y_pred = []
    for n_layers in range(1,3):
        ##########################################
        ### Creating Neural Network with Keras ###
        ##########################################
        print(f"Training with depth {n_layers}")
        weight_shapes = {"weights": (n_layers,n_qubits,3)}

        dev = qml.device("default.qubit", wires=n_qubits)
        qnn = qml.QNode(qnode_circular_entangling, dev, interface="tf")
        q_layer = qml.qnn.KerasLayer(qnn, weight_shapes, output_dim=n_qubits)

        Activation=tf.keras.layers.Activation(tf.keras.activations.linear)
        output_layer = tf.keras.layers.Dense(prev,kernel_initializer='normal')

        opt = tf.optimizers.Adamax(learning_rate=0.1)

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
                                , verbose=0
                                , validation_data=(X_val, y_val)
                                , callbacks = [es, re])

        #################
        ### Loss Plot ###
        #################
        plot_history(history_model, n_layers)

        ##################
        ### Prediction ###
        ##################
        y_pred = model.predict(X_test_scaled,verbose=0)
        list_y_pred.append(y_pred)
        mean_predictions, mean_error_normal, mean_error_left_normal, mean_error_right_normal = get_mean_left_right_error_interval(model, scaler_x, X_val, y_val, y_test, y_pred)
        plot_prediction_versus_observed(n_layers, city, height, y_test, y_pred, mean_error_normal)
        
        print(f"Wilcoxon test Depth {n_layers}\n")
        verify_distribution_wilcoxtest(y_test[:,0],y_pred[:,0], 0.05)
        print("\n#########\n")

    #####################
    ### Data Analysis ###
    #####################
    print("Len list_y_pred", len(list_y_pred))
    #print(list_y_pred)
    all_analysis = quantitative_analysis(y_test, list_y_pred)
    print(all_analysis)
    print("\n#########\n")

    path = os.path.abspath(os.path.join(os.getcwd(), 'analysis'))
    filename = f"metrics-public.txt"
    all_analysis.to_csv(os.path.join(path,filename))

if __name__ == "__main__":
    main()
