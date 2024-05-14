# samples.py from http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/classification.html used for file reading

import util
import numpy as np
import zipfile
import os
import time

## Constants
DATUM_WIDTH_FACE = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

class Perceptron:
 
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
        items.append(Perceptron(data, DATUM_WIDTH, DATUM_HEIGHT))
        count =+ 1
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

# function used to train the data and get the final weights which will be used on the actual data  
def train_face(n):

  epochs = 1
  data = loadDataFile('data/facedata/facedatatrain', 451, 60, 70)
  labels = loadLabelsFile('data/facedata/facedatatrainlabels', 451)
  weights = np.random.uniform(low= -1, high= 1, size=(70, 60))
  num_samples = int(n * 451)

  final_accuracies = []
  final_std = []

  bias = np.random.uniform(low = -1, high = 1)

  for epoch in range(epochs):
    images_used = set()
    for _ in range(num_samples):

      idx = np.random.randint(0, 451)

      while idx in images_used:
          idx = np.random.randint(0, 451)

      images_used.add(idx)
      sample = data[idx].getPixels()

      total_sum = bias + np.sum(sample * weights)

      label = labels[idx]

      if total_sum > 0 and label == 0:
          bias -= 1
          weights -= sample

      elif total_sum < 0 and label == 1:
          bias += 1
          weights += sample

  ############## END OF TRAINING MODEL CODE FOR FACE ###################            
      
    # now test the test data to see how accurate it is    
    data_test = loadDataFile('data/facedata/facedatavalidation', 301, 60, 70)
    labels_test = loadLabelsFile('data/facedata/facedatavalidationlabels', 301)
    
    accuracies = []

    for idx in range(301):

      sample = data_test[idx].getPixels()

      total_sum = bias + np.sum(sample * weights)

      label = labels_test[idx]

      if (total_sum > 0 and label == 1):
        accuracies.append(1)
      elif (total_sum < 0 and label == 0):
        accuracies.append(1)
      else:
        accuracies.append(0)

    print(f'Accuracy Change at Epoch {epoch}: {np.average(accuracies)}')
    final_accuracies.append(np.average(accuracies))
    final_std.append(np.std(accuracies))  
  
  return np.average(final_accuracies), np.std(final_std)  

def test_face():
   
  data_test = loadDataFile('data/facedata/facedatavalidation', 301, 60, 70)
  labels_test = loadLabelsFile('data/facedata/facedatavalidationlabels', 301)

  file_path = "perceptron_face_weightsbias.txt"

  # Initialize variables to store weights array and bias value
  weights = None
  bias = None

  # Read the contents of the text file
  with open(file_path, "r") as file:
      lines = file.readlines()

  # Extract the weights array
  weights_values = []
  for line in lines[:-1]:  # Exclude the last line which contains the bias value
      row_values = line.strip().split()
      weights_values.append([float(val) for val in row_values])
  weights = np.array(weights_values)

  # Extract the bias value
  bias = float(lines[-1])
    
  accuracies = []

  for idx in range(301):

    sample = data_test[idx].getPixels()

    total_sum = bias + np.sum(sample * weights)

    label = labels_test[idx]

    if (total_sum > 0 and label == 1):
      accuracies.append(1)
    elif (total_sum < 0 and label == 0):
      accuracies.append(1)
    else:
      accuracies.append(0)

  return np.average(accuracies), np.std(accuracies)

# Move validation loop in epoch loop
def train_digit(n):
  
  epochs = 10
  data = loadDataFile('data/digitdata/trainingimages', 5000, 28, 28)
  labels = loadLabelsFile('data/digitdata/traininglabels', 5000)
  weights = np.random.randint(low=-400, high=400, size=(10, 28, 28))
  num_samples = int(n * 5000)
  accuracies = []
  
  final_accuracies = []
  final_std = []
  bias = np.random.randint(low=-200, high=200, size=10)

  for epoch in range(epochs):
    images_used = set()
    for _ in range(num_samples):
      
      # random idx from the training data
      idx = np.random.randint(0, 5000)
      
      while idx in images_used:
        idx = np.random.randint(0, 5000)
      
      images_used.add(idx)

      # the image at that index in the training data
      image = np.array(data[idx].getPixels())

      # Compute the total sum for each digit using NumPy vectorization
      total_sums = np.sum(weights * image, axis=(1, 2)) + bias

      # Get the predicted digit and the actual digit
      predicted_digit = np.argmax(total_sums)
      real_digit = labels[idx]

      # Update weights and bias if prediction is incorrect
      if predicted_digit != real_digit:
        bias[predicted_digit] -= 1
        bias[real_digit] += 1
        weights[predicted_digit] -= image
        weights[real_digit] += image

############## END OF TRAINING MODEL CODE FOR DIGIT ################### 

  # now test the test data to see how accurate it is    
    data_test = loadDataFile('data/digitdata/validationimages', 1000, 28, 28)
    labels_test = loadLabelsFile('data/digitdata/validationlabels', 1000)

    accuracies = []
    
    for idx in range(1000):

        image = np.array(data_test[idx].getPixels())

        # Compute the total sum for each digit using NumPy vectorization
        total_sums = np.sum(weights * image, axis=(1, 2)) + bias

        # Get the predicted digit and the actual digit
        predicted_digit = np.argmax(total_sums)
        real_digit = labels_test[idx]

        if real_digit == predicted_digit:
          accuracies.append(1)
        else:
          accuracies.append(0)

    print(f'Accuracy Change at Epoch {epoch}: {np.average(accuracies)}')
    final_accuracies.append(np.average(accuracies))
    final_std.append(np.std(accuracies))

  return np.average(final_accuracies), np.std(final_std) 

def test_digit():

  file_path = "perceptron_digit_weightsbias.txt"
   
  weights = None
  bias = None

# Read the contents of the text file
  with open(file_path, "r") as file:
      lines = file.readlines()

  # Extract the weights array
  weights_values = []
  current_matrix = []
  for line in lines:
      if line.strip() == "":
          weights_values.append(current_matrix)
          current_matrix = []
      else:
          row_values = line.strip().split()
          current_matrix.append([int(val) for val in row_values])
  weights = np.array(weights_values)

  # Extract the bias array
  bias_values = lines[-1].strip().split()
  bias = np.array([int(val) for val in bias_values])

  data_test = loadDataFile('data/digitdata/validationimages', 1000, 28, 28)
  labels_test = loadLabelsFile('data/digitdata/validationlabels', 1000)

  accuracies = []
  
  for idx in range(1000):

      image = np.array(data_test[idx].getPixels())

      # Compute the total sum for each digit using NumPy vectorization
      total_sums = np.sum(weights * image, axis=(1, 2)) + bias

      # Get the predicted digit and the actual digit
      predicted_digit = np.argmax(total_sums)
      real_digit = labels_test[idx]

      if real_digit == predicted_digit:
        accuracies.append(1)
      else:
        accuracies.append(0)

  return np.average(accuracies), np.std(accuracies)

# Testing
def savedata_forall():

  averages, stds, times = [], [], []

  values = [i / 10 for i in range(1, 11)]

  for value in values:
    start_time = time.time()
    average, std = train_face(value)
    averages.append(average)
    stds.append(std)
    times.append(time.time() - start_time)

  print(averages)
  print(stds)  
  print(times)

def train_data():
  average, std = train_face(1)
  print(average, std)

def test_networks():
  average, std = test_face()
  print(average, std)


if __name__ == "__main__":
  test_networks()



