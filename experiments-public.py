import os
import pennylane as qml
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adamax
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils.models import QuantumNeuralNetwork, EarlyStopper
from utils.datasets import WindDataset
from utils.statistics import quantitative_analysis, get_mean_left_right_error_interval, verify_distribution_wilcoxtest, model_predict

def plot_history(history, n_layers):
    plt.figure(figsize=(14,5), dpi=320, facecolor='w', edgecolor='k')
    plt.title(f"Loss for depth {n_layers}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(history['loss'], label="Training Loss")
    plt.plot(history['val_loss'], label="Validation Loss")
    plt.xticks(range(0, len(history['loss'])+1, 5))
    plt.legend()
    plt.grid()

    path = os.path.abspath(os.path.join(os.getcwd(), 'plots'))
    filename = f"loss-history-public-{n_layers}.png"
    plt.savefig(os.path.join(path,filename))

    values = np.array([list(range(1, len(history['loss'])+1)), history['loss'], history['val_loss']])
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


def load_table(path, prev):
    X=pd.read_csv(path)
    X.dropna(axis=0,how='any',inplace=True)
    #X = X.loc[X['Year'].isin([2016])]

    # Using only the target column and removing the first index to predict the next target
    y = X[:].drop(X.index[0])

    # Remove the last line of X because the predicted Y will not have an extra line
    X = X.iloc[:-prev[-1],:]

    for i in prev:
        y[f'Prev {i} hour'] = y.loc[:,"Temperature"].shift(-(i-1))
    if prev[-1] == 1:
        y= y.iloc[:, -1:]
    else:
        y= y.iloc[:-(prev[-1]-1), -len(prev):]

    return X, y.values


def main():
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    # Decide which device we want to run on
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)

    ######################
    ### Importing Data ###
    ######################
    prevision_window = [1,2,3,4,5,6]
    print("\nLoading Datasets\n")

    path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    filename = "WeatherForecasting.csv"
    dataroot = os.path.join(path,filename)

    print(f"importing data from {dataroot}")
    X_all,y_all = load_table(dataroot, prevision_window)

    # Splitting dataframe
    X_train_val = X_all.iloc[:int(X_all.shape[0]*.7),:]
    y_train_val = y_all[:int(X_all.shape[0]*.7)]
    X_test      = X_all.iloc[int(X_all.shape[0]*.7):,:]
    y_test      = y_all[int(X_all.shape[0]*.7):]

    print(f"There are {X_train_val.shape[1]} features and {X_train_val.shape[0]} instances in Train-Val set")
    print(X_train_val.head())
    print(f"There are {X_test.shape[1]} features and {X_test.shape[0]} instances in Test set")
    print(X_test.head())
    
    print("Size y_train_val", len(y_train_val),"\n", y_train_val[:5])
    print("Size y_test", len(y_test),"\n", y_test[:5])
    print("\n#########\n")

    ####################
    ### Scaling Data ###
    ####################
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    #scaler_x = StandardScaler()
    Xs_train_val = scaler_x.fit_transform(X_train_val)
    Xs_test = scaler_x.transform(X_test)

    #####################################
    ### Splitting Train and Test sets ###
    #####################################
    print("\nSplitting Train Data\n")
    train_ratio = 0.8
    Xs_train, Xs_val, y_train, y_val = train_test_split(Xs_train_val, y_train_val, test_size=1 - train_ratio)

    print(f"There are {Xs_train.shape[1]} features and {Xs_train.shape[0]} instances in Train set")
    print(f"There are {Xs_val.shape[1]} features and {Xs_val.shape[0]} instances in Val set")
    print(f"There are {Xs_test.shape[1]} features and {Xs_test.shape[0]} instances in Test set")

    trainset = WindDataset(X = Xs_train
                      , y = y_train
                     )
    valset = WindDataset(X = Xs_val
                          , y = y_val
                         )
    testset = WindDataset(X = Xs_test
                          , y = y_test
                         )

    trainloader = DataLoader(trainset
                            , batch_size = 64
                            , num_workers = 8
                            )
    valloader = DataLoader(valset
                            , batch_size = len(valset)
                            , num_workers = 8
                            )
    testloader = DataLoader(testset
                            , batch_size = len(testset)
                            , num_workers = 8
                            )

    print("\n#########\n")

    n_features = Xs_train.shape[1]
    n_qubits = n_features
    n_layers = 2
    print(f"Circuit size: {n_qubits} qubits")

    list_y_pred = []
    for n_layers in range(1,3):
        ##########################################
        ### Creating Neural Network with Keras ###
        ##########################################
        print(f"Training with depth {n_layers}")
        qnn = QuantumNeuralNetwork(len(prevision_window), n_qubits, 'default.qubit', n_layers, device).to(device)
        print(f"QNN:\n{qnn}")
        print(f"Num Parameters: {sum(p.numel() for p in qnn.parameters())}")

        ##################
        # Printing Model #
        ##################
        sampl_input = torch.randn(n_qubits, device=device)
        sampl_weights = torch.rand(n_layers,n_qubits,3)

        print(qml.draw(qnn._quantum_circuit, expansion_strategy="device")(sampl_input, sampl_weights))

        ######################
        ### Training Model ###
        ######################
        lrate = 0.1
        criterion = nn.MSELoss()
        optimizer = Adamax(qnn.parameters(), lr=lrate)
        early_stopper = EarlyStopper(patience=3, min_delta=0.01)

        num_epochs = 30 # Number of training epochs
        results = []
        loss_list = []
        val_loss_list = []
        test_input = next(iter(testloader))[0].to(device)

        for epoch in range(num_epochs):
            print('Epoch: ', epoch)
            loss_epoch = []
            val_loss_epoch = []
            # For each batch in the dataloader
            for i, (data, target) in enumerate(trainloader):
                #print(f"Step {i}")
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                out = qnn(data)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                loss_epoch.append(loss.item())
            loss_epoch_mean = np.mean(loss_epoch)
            print('loss:', loss_epoch_mean)
            loss_list.append(loss_epoch_mean)
            # Validation
            with torch.no_grad():
                for i, (val_data, val_target) in enumerate(valloader):
                    val_data = val_data.to(device)
                    val_target = val_target.to(device)

                    val_out = qnn(val_data)
                    val_loss = criterion(val_out, val_target)

                    val_loss_epoch.append(val_loss.item())
                val_loss_mean = np.mean(val_loss_epoch)
                print('val_loss:', val_loss_mean)
                val_loss_list.append(val_loss_mean)
            if early_stopper.early_stop(val_loss):             
                    break

        #################
        ### Loss Plot ###
        #################
        history_model = {"loss": loss_list, "val_loss": val_loss_list}
        plot_history(history_model, n_layers)

        ##################
        ### Prediction ###
        ##################
        y_pred = model_predict(qnn, test_input)
        list_y_pred.append(y_pred)
        mean_predictions, mean_error_normal, mean_error_left_normal, mean_error_right_normal = get_mean_left_right_error_interval(qnn, scaler_x, valloader, y_test, y_pred, device)
        plot_prediction_versus_observed(n_layers, y_test, y_pred, mean_error_normal)

        print(f"Wilcoxon test Depth {n_layers}\n")
        verify_distribution_wilcoxtest(y_test[:,0],y_pred[:,0], 0.05)
        print("\n#########\n")

    #####################
    ### Data Analysis ###
    #####################
    print("Len list_y_pred", len(list_y_pred))
    #print(list_y_pred)
    for depth, predito in enumerate(list_y_pred):
        all_analysis = quantitative_analysis(y_test[:-10], predito[:-10])
        print(all_analysis)
        print("\n#########\n")

        path = os.path.abspath(os.path.join(os.getcwd(), 'analysis'))
        filename = f"metrics-public-depth-{depth}.txt"
        all_analysis.to_csv(os.path.join(path,filename))

if __name__ == "__main__":
    main()
