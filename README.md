Quick implementation of binary perceptron and four basic logic gates:

* AND gate
* NAND gate
* OR gate
* NOR gate 

``` Python

class Neuron:
  
  def __init__(self, input : list[int], weights : list[int],  T : int):
    
    self.input = np.array(input)
    self.weights = np.array(weights)
    self.T = T
    
  def forward(self) -> int:
    
    """
    Calculates output value of single neuron for defined input,
    weight and treshold value using binary activation function.
    """
    weighted_sum = sum(self.input * self.weights)
    return 0 if weighted_sum < self.T else 1


class LogicGate:
  
  def __init__(self, input : list[int]):
    self.input = np.array(input)

  def and_gate(self) -> int:
    """
    Calculates output value for and gate. 
    """
    n1_val = Neuron(input = self.input,
                    weights = [1, 1],
                    T = 2).forward()
    return n1_val
  
  def or_gate(self) -> int:   
    """
    Calculates output value for or gate. 
    """   
    n1_val = Neuron(input = self.input,
                    weights = [1, 1],
                    T = 1).forward()
    return n1_val

  def nor_gate(self) -> int:
    """
    Calculates output value for nor gate. 
    """
    n1_val = Neuron(input = self.input,
                    weights = [1, 1],
                    T = 1).forward()
    
    n2_val = Neuron(input = np.array([n1_val]),
                    weights = [-1],
                    T = 0).forward()
    return n2_val

  def nand_gate(self) -> int:
    """
    Calculates output value for nand gate.
    """
    n1_val = Neuron(input = self.input[0],
                    weights = [-1],
                    T = 0).forward()
    
    n2_val = Neuron(input = self.input[1],
                    weights = [-1],
                    T = 0).forward()
    
    n3_val = Neuron(input = [n1_val, n2_val],
                    weights = [1,1],
                    T = 1).forward()
    return n3_val


```
