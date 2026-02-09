# EXPERIMENT 1 - Developing a Neural Network Regression Model
## NAME : SAJEN MURALI
## REGISTRATION NUMBER : 212223220089

## AIM :
To develop a neural network regression model for the given dataset.

## THEORY :
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features.
The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model :
<img width="1820" height="1017" alt="Screenshot 2026-02-02 094607EXP1" src="https://github.com/user-attachments/assets/91177b10-6ef2-428c-b60f-6fbbf3c926d2" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM :

### Name: DHAMINI S

### Register Number: 212224040064

```
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(ai_brain(X_train),y_train)
    loss.backward()
    optimizer.step()


    ai_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```

## Dataset Information :

<img width="252" height="287" alt="d" src="https://github.com/user-attachments/assets/73a8a163-c720-4f8d-a79d-267a6344c2dd" />



## OUTPUT :

<img width="362" height="230" alt="d2" src="https://github.com/user-attachments/assets/845edd7b-43d8-44f0-95b4-6a1f60e25401" />

<img width="258" height="35" alt="d5" src="https://github.com/user-attachments/assets/87477645-78ec-4197-96e1-cc68fefe4b7a" />



## Training Loss Vs Iteration Plot :

<img width="727" height="568" alt="graph" src="https://github.com/user-attachments/assets/226690f1-76e8-41fe-81da-3329689bb83f" />


## New Sample Data Prediction :
<img width="422" height="37" alt="d1" src="https://github.com/user-attachments/assets/3d31855a-a7cf-4293-bd08-71f0cb173649" />



## RESULT : 
Thus, a neural network regression model was successfully developed and trained using PyTorch.

