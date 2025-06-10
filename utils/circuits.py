import pennylane as qml

def qnode_circular_entangling(inputs, weights):
    # weights: (n_layers,n_qubits,3)
    n_layers = len(weights)
    n_qubits = len(weights[0])
    
    ###############
    # Feature Map #
    ###############
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)
    qml.templates.AngleEmbedding(inputs, rotation='Y', wires=range(n_qubits))
    
    ##########
    # Ansatz #
    ##########
    for k in range(n_layers):
        # Entangling Layer
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i,i+1])
        qml.CNOT(wires=[n_qubits-1,0])
        
        # Variational Layer
        for i in range(len(weights[k])):
            qml.Rot(*weights[k][i],wires=i)
    
    ###############
    # Measurement #
    ###############
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

def qnode_parallel_entangling(inputs, weights):
    # weights: (n_layers,n_qubits,3)
    n_layers = len(weights)
    n_qubits = len(weights[0])
    
    ###############
    # Feature Map #
    ###############
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)
    qml.templates.AngleEmbedding(inputs, rotation='Y', wires=range(n_qubits))
    
    ##########
    # Ansatz #
    ##########
    for k in range(n_layers):
        # Entangling Layer
        for i in range(0, n_qubits - 1, 2): 
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, n_qubits - 1, 2):  
            qml.CNOT(wires=[i, i + 1])
        
        # Variational Layer
        for i in range(len(weights[k])):
            qml.Rot(*weights[k][i],wires=i)
    
    ###############
    # Measurement #
    ###############
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
