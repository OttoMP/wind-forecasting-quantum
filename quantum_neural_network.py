import pennylane as qml
import numpy as np

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

#dev = qml.device('lightning.qubit', wires=n_qubits)
#dev = qml.device('default.qubit', wires=5)
dev = qml.device('lightning.gpu', wires=5)
@qml.qnode(dev)
def qnode_strong_entangling(inputs, weights):
    # weights: (n_layers,n_qubits,3)
    # len(weights) == n_layers
    # len(weights[0]) == num_qubits
    H_layer(len(weights[0]))
    Data_AngleEmbedding_layer(inputs, len(weights[0]))
    for k in range(len(weights)):
        strong_entangling_layer(len(weights[0]))
        ROT_layer(weights[k])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(len(weights[0]))]

#dev = qml.device('lightning.gpu', wires=n_qubits)
#dev = qml.device('lightning.qubit', wires=n_qubits)
dev = qml.device('default.qubit', wires=5)
@qml.qnode(dev, interface="tensorflow")
def qnode_entangling(inputs, weights):
    # weights: (n_layers,n_qubits,3)
    # len(weights) == n_layers
    # len(weights[0]) == num_qubits
    H_layer(len(weights[0]))
    Data_AngleEmbedding_layer(inputs, len(weights[0]))
    for k in range(len(weights)):
        entangling_layer(len(weights[0]))
        ROT_layer(weights[k])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(len(weights[0]))]

def draw_circuit():
    n_qubits = 5
    print(f"Serão necessários {n_qubits} qubits")
    n_layers = 1
    weight_shapes = {"weights_1": (n_layers,n_qubits,3)}

    sampl_weights = np.random.uniform(low=0, high=np.pi, size=weight_shapes["weights_1"])
    print(sampl_weights)
    print(len(sampl_weights))
    print(len(sampl_weights[0]))
    print(len(sampl_weights[0][0]))

    sampl_input = np.random.uniform(low=0, high=np.pi, size=(n_qubits,))
    print(qml.draw(qnode_entangling, expansion_strategy="device")(sampl_input, sampl_weights))


if __name__ == "__main__":
    draw_circuit()
