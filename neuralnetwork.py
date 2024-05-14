import util
import numpy as np
import zipfile
import os
import math
import time

## Constants
DATUM_WIDTH_FACE = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

class Neural_Network:
 
  def __init__(self, data,width,height):
    """
    Create a new datum from file input (standard MNIST encoding).
    """
    DATUM_HEIGHT = height
    DATUM_WIDTH=width
    self.height = DATUM_HEIGHT
    self.width = DATUM_WIDTH
    self.weight_faces = None
    if data == None:
      data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)] 
    self.pixels = data
    
  def getPixel(self, column, row):
    """
    Returns the value of the pixel at column, row as 0, or 1.
    """
    return self.pixels[column][row]
      
  def getPixels(self):
    """
    Returns all pixels as a list of lists.
    """
    return self.pixels    
      
# Data processing, cleanup and display functions  
def loadDataFile(filename, n, width, height):
    """
    Reads n data images from a file and returns a list of Datum objects.
    
    (Return less than n items if the end of file is encountered).
    """
    DATUM_WIDTH = width
    DATUM_HEIGHT = height
    fin = readlines(filename)
    fin.reverse()
    items = []
    count = 0
    for i in range(n):
        data = []
        for j in range(height):
            # Read a line from the file
            line = fin.pop()
            # Convert symbols to 0s and 1s
            #print(line)
            #print(list(map(convertToInteger, line)))
            data.append(list(map(convertToInteger, line)))
        if len(data[0]) < DATUM_WIDTH - 1:
            # We encountered the end of the file
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(Neural_Network(data, DATUM_WIDTH, DATUM_HEIGHT))
        count = + 1
    return items


def readlines(filename):
  "Opens a file or reads it from the zip archive data.zip"
  if(os.path.exists(filename)): 
    return [l[:-1] for l in open(filename).readlines()]
  else: 
    z = zipfile.ZipFile('data.zip')
    return z.read(filename).split('\n')
    
def loadLabelsFile(filename, n):
  """
  Reads n labels from a file and returns a list of integers.
  """
  fin = readlines(filename)
  labels = []
  for line in fin[:min(n, len(fin))]:
    if line == '':
        break
    labels.append(int(line))
  return labels  
    
def IntegerConversionFunction(character):
  """
  Helper function for file reading.
  """
  if(character == ' '):
    return 0
  elif(character == '+'):
    return 1
  elif(character == '#'):
    return 2    

def convertToInteger(data):
  """
  Helper function for file reading.
  """
  if type(data) != type([]):
    return IntegerConversionFunction(data)
  else:
    return map(convertToInteger, data)

def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(output):
    return sigmoid_activation(output) * (1 - sigmoid_activation(output))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_derivative(x):
    exp_x = np.exp(x)
    exp_x_sum = np.sum(exp_x, axis=1, keepdims=True)
    softmax_x = exp_x / exp_x_sum
    return softmax_x * (1 - softmax_x)

def backpropagate(sample, predicted, actual, hidden_layer_values, output_weights, hidden_weights):
    
    # Calculate the error at the output layer
    hidden_layer_values = np.array(hidden_layer_values)
    learning_rate = 0.01
    error = predicted - actual
    sample = sample.reshape(1, -1)

    # calculate the output gradient
    output_gradient = hidden_layer_values * error


    hidden_out = ((output_weights.T * error) * sigmoid_derivative(hidden_layer_values))
    hidden_out = hidden_out.reshape((1000, 1))
    hidden_gradient = hidden_out * sample
    
    # update the weights
    output_weights -= learning_rate * output_gradient
    hidden_weights -= learning_rate * hidden_gradient
    

    return output_weights, hidden_weights

def backpropagate_digit(sample, predicted, actual, hidden_layer_values, output_weights, hidden_weights):
    # Calculate the error at the output layer
    hidden_layer_values = np.array(hidden_layer_values)
    hidden_layer_values = hidden_layer_values.reshape(256, 1)
    learning_rate = 0.02
    error = predicted - actual
    error = error.reshape(1, 10)
    sample = sample.reshape(1, -1)

    # Calculate the output gradient
    output_gradient = hidden_layer_values * error

    # Calculate the hidden gradient
    error = error.reshape(10, 1)
    hidden_out = (np.dot(output_weights.T, error) * sigmoid_derivative(hidden_layer_values))
    hidden_gradient = hidden_out * sample

    # Update the weights
    output_weights -= learning_rate * output_gradient.reshape((10, 256))
    hidden_weights -= learning_rate * hidden_gradient

    return output_weights, hidden_weights
    

def nn_face(n):
  
  epochs = 10
  data = loadDataFile('data/facedata/facedatatrain', 451, 60, 70)
  labels = loadLabelsFile('data/facedata/facedatatrainlabels', 451)
  hidden_weights = np.random.uniform(low=-1, high=1, size=(1000, 4200))  # we need different weights for each of the nodes on the hidden layer
  output_weights = np.random.uniform(low=-1, high=1, size=1000)
  bias = 1

  num_samples = int(n * 451)

  final_accuracies = []
  final_std = []

  for epoch in range(epochs):
    
    images_used = set()
    for _ in range(num_samples):
      
      hidden_layer_values = []

      idx = np.random.randint(0, 451)

      while idx in images_used:
        idx = np.random.randint(0, 451)

      images_used.add(idx)
      sample = data[idx]
      sample = np.array(sample.getPixels()).flatten()

      total_sum = 0
      for node in range(1000):
          total_sum = np.dot(hidden_weights[node], sample) 
          hidden_layer_values.append(sigmoid_activation(total_sum + bias))
      
      final_output = np.sum(output_weights * hidden_layer_values)
      
      predicted = sigmoid_activation(final_output + bias)
      actual = labels[idx]

      if predicted < 0.5 and actual == 1 :
          #print("About to backpropagate.")
          output_weights, hidden_weights = backpropagate(sample, predicted, actual, hidden_layer_values, output_weights, hidden_weights)
          #print("Finished backpropagating.")
      elif predicted > 0.5 and actual == 0:
          #print("About to backpropagate.")
          output_weights, hidden_weights= backpropagate(sample, predicted, actual, hidden_layer_values, output_weights, hidden_weights)
          #print("Finished backpropagating.")          

  # now test the test data to see how accurate it is    
    data_test = loadDataFile('data/facedata/facedatavalidation', 301, 60, 70)
    labels_test = loadLabelsFile('data/facedata/facedatavalidationlabels', 301)
    
    accuracies = []

    for idx in range(301):
        hidden_layer_values = []

        sample = data_test[idx]

        sample = np.array(sample.getPixels()).flatten()

        total_sum = 0
        for node in range(1000):
            total_sum = np.dot(hidden_weights[node], sample) 
            hidden_layer_values.append(sigmoid_activation(total_sum + bias))
        
        final_output = np.sum(output_weights * hidden_layer_values)
        
        predicted = sigmoid_activation(final_output + bias)
        actual = labels_test[idx]

        if (predicted > 0.5 and actual == 1) or (predicted < 0.5 and actual == 0): 
          accuracies.append(1)
        else:
          accuracies.append(0)

    print(f'Accuracy Change at Epoch {epoch}: {np.average(accuracies)}')
    final_accuracies.append(np.average(accuracies))
    final_std.append(np.std(accuracies))

  return np.average(final_accuracies), np.std(final_std)

def nn_face_test():
  
  data_test = loadDataFile('data/facedata/facedatavalidation', 301, 60, 70)
  labels_test = loadLabelsFile('data/facedata/facedatavalidationlabels', 301)

  file_path = "nn_face_weightsbias.txt"

  # Initialize variables to store the arrays
  hidden_weights = None
  output_weights = None
  bias = None

  # Read the contents of the text file
  with open(file_path, "r") as file:
      lines = file.readlines()

  # Extract hidden weights array
  hidden_weights_values = []
  for line in lines[:-1]:  # Exclude the last line which contains the bias value
      row_values = line.strip().split()
      hidden_weights_values.append([float(val) for val in row_values])
  hidden_weights = np.array(hidden_weights_values)

  # Extract output weights array
  output_weights_values = lines[-2].strip().split()
  output_weights = np.array([float(val) for val in output_weights_values])

  # Extract bias value
  bias = float(lines[-1])
  
  accuracies = []

  for idx in range(301):
      hidden_layer_values = []

      sample = data_test[idx]


      sample = np.array(sample.getPixels()).flatten()

      total_sum = 0
      for node in range(1000):
          total_sum = np.dot(hidden_weights[node], sample) 
          hidden_layer_values.append(sigmoid_activation(total_sum + bias))
      
      final_output = np.sum(output_weights * hidden_layer_values)
      
      predicted = sigmoid_activation(final_output + bias)
      actual = labels_test[idx]

      if (predicted > 0.5 and actual == 1) or (predicted < 0.5 and actual == 0): 
        accuracies.append(1)
      else:
        accuracies.append(0)

  return np.average(accuracies), np.std(accuracies)  


def nn_digit(n):
  
  epochs = 10
  data = loadDataFile('data/digitdata/trainingimages', 451, 28, 28)
  labels = loadLabelsFile('data/digitdata/traininglabels', 451)
  hidden_weights = np.random.uniform(low=-1, high=1, size=(256, 784))  # we need different weights for each of the nodes on the hidden layer
  output_weights = np.random.uniform(low=-1, high=1, size=(10, 256))
  output_weights = output_weights.reshape((10, 256))
  bias = 1

  num_samples = int(n * 451)

  final_accuracies = []
  final_std = []

  for epoch in range(epochs):
    images_used = set()
    for _ in range(num_samples):
      
      hidden_layer_values = [] 
      predicted_arr = np.zeros(10)
      actual_arr = np.zeros(10)
      
      idx = np.random.randint(0, 451)

      while idx in images_used:
        idx = np.random.randint(0, 451)

      images_used.add(idx)

      sample = data[idx]
      sample = np.array(sample.getPixels()).flatten()

      for node in range(256):
        total_sum = np.dot(hidden_weights[node], sample) 
        hidden_layer_values.append(sigmoid_activation(total_sum + bias))

      for digit in range(10):
        final_output = sigmoid_activation(np.sum(output_weights[digit] * hidden_layer_values) + bias) 
        predicted_arr[digit] = final_output

      actual_digit = labels[idx]
      actual_arr[actual_digit] = 1

      predicted_digit =  np.argmax(predicted_arr)

      if predicted_digit != actual_digit:
         output_weights, hidden_weights= backpropagate_digit(sample, predicted_arr, actual_arr, hidden_layer_values, output_weights, hidden_weights)


    data_test = loadDataFile('data/digitdata/validationimages', 1000, 28, 28)
    labels_test = loadLabelsFile('data/digitdata/validationlabels', 1000)

    accuracies = []
    
    for idx in range(1000):

        hidden_layer_values = [] 
        predicted_arr = np.zeros(10)
        actual_arr = np.zeros(10)
        
        idx = np.random.randint(0, 451)
        sample = data_test[idx]
        sample = np.array(sample.getPixels()).flatten()

        for node in range(256):
          total_sum = np.dot(hidden_weights[node], sample) 
          hidden_layer_values.append(sigmoid_activation(total_sum + bias))

        for digit in range(10):
          final_output = sigmoid_activation(np.sum(output_weights[digit] * hidden_layer_values) + bias)
          predicted_arr[digit] = final_output

        actual_digit = labels_test[idx]
        actual_arr[actual_digit] = 1

        predicted_digit =  np.argmax(predicted_arr)

        if predicted_digit == actual_digit:
          accuracies.append(1)
        else:
          accuracies.append(0)
    
    print(f'Accuracy Change at Epoch {epoch}: {np.average(accuracies)}')
    final_accuracies.append(np.average(accuracies))
    final_std.append(np.std(accuracies))

  return np.average(final_accuracies), np.std(final_std)  

def nn_digit_test():
   
  data_test = loadDataFile('data/digitdata/validationimages', 1000, 28, 28)
  labels_test = loadLabelsFile('data/digitdata/validationlabels', 1000)

  file_path = "nn_digit_weightsbias.txt"

  hidden_weights = None
  output_weights = None
  bias = None

  # Read the contents of the text file
  with open(file_path, "r") as file:
      lines = file.readlines()

  # Extract hidden weights array
  hidden_weights_values = []
  for line in lines[:256]:  # Assuming hidden weights array has 256 rows
      row_values = line.strip().split()
      hidden_weights_values.append([float(val) for val in row_values])
  hidden_weights = np.array(hidden_weights_values)

  # Extract output weights array
  output_weights_values = []
  for line in lines[256:266]:  # Assuming output weights array has 10 rows
      row_values = line.strip().split()
      output_weights_values.append([float(val) for val in row_values])
  output_weights = np.array(output_weights_values)

    # Extract bias value
  bias = float(lines[-1])

  accuracies = []
  
  for idx in range(1000):

      hidden_layer_values = [] 
      predicted_arr = np.zeros(10)
      actual_arr = np.zeros(10)
      
      idx = np.random.randint(0, 451)
      sample = data_test[idx]
      sample = np.array(sample.getPixels()).flatten()

      for node in range(256):
        total_sum = np.dot(hidden_weights[node], sample) 
        hidden_layer_values.append(sigmoid_activation(total_sum + bias))

      for digit in range(10):
        final_output = sigmoid_activation(np.sum(output_weights[digit] * hidden_layer_values) + bias)
        predicted_arr[digit] = final_output

      actual_digit = labels_test[idx]
      actual_arr[actual_digit] = 1

      predicted_digit =  np.argmax(predicted_arr)

      if predicted_digit == actual_digit:
        accuracies.append(1)
      else:
        accuracies.append(0)

  return np.average(accuracies), np.std(accuracies)
    
def _test():
 
 averages, stds, times = [], [], []

 values = [i / 10 for i in range(1, 11)]

 for value in values:
  start_time = time.time()
  average, std = nn_digit(value)
  averages.append(average)
  stds.append(std)
  times.append(time.time() - start_time)

 print(averages)
 print(stds)  
 print(times)

def train_data():
  average, std = nn_digit(.5)
  print(average, std)

def test_networks():
  average, std = nn_digit_test()
  print(average, std)
 

if __name__ == "__main__":
  test_networks()  