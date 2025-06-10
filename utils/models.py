import torch
import torch.nn as nn
import pennylane as qml

def quantum_circuit(inputs, weights):
    # unpack weights
    qdepth = weights.size(dim=0)
    nqubits = weights.size(dim=1)
    
    #############
    # Embedding #
    #############
    for i in range(nqubits):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i], wires=i)

    ##########
    # Ansatz #
    ##########
    for k in range(qdepth):
        # Entangling
        for i in range(0, nqubits - 1, 2): 
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, nqubits - 1, 2):  
            qml.CNOT(wires=[i, i + 1])
        # Rotation
        for y in range(nqubits):
            #qml.RY(weights[k][y], wires=y)
            qml.Rot(*weights[k][y], wires=y)

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(nqubits)]

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, forecast_window_size, num_qubits, QML_device, num_layers, torch_device): 
        super(QuantumNeuralNetwork, self).__init__() 
        self.forecast_window_size = forecast_window_size
        
        self.device = torch_device
        penny_dev = qml.device(QML_device, wires = num_qubits)
        
        qnode = qml.QNode(quantum_circuit, penny_dev, interface='torch', diff_method="best")
        self._quantum_circuit = qml.transforms.broadcast_expand(qnode)

        q_weights_shape = {'weights':(num_layers, num_qubits, 3)}
        self.hidden_quantum_layer = qml.qnn.TorchLayer(self._quantum_circuit, q_weights_shape)        
        
        self.output_classical_layer = torch.nn.Linear(num_qubits, forecast_window_size) 
    
    def forward(self, batch):
        batch_output = []
        for y in batch:
            y = self.hidden_quantum_layer(y).to(self.device)
            y = self.output_classical_layer(y) 
            batch_output.append(y)

        return torch.vstack(batch_output) # batch_output: [Batch, Output length]
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False