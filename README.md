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

<img width="197" height="282" alt="image" src="https://github.com/user-attachments/assets/66b4d73a-7554-4e7c-a254-284266224398" />




## OUTPUT :

<img width="468" height="237" alt="image" src="https://github.com/user-attachments/assets/7113a799-d626-4f49-903e-e9386bbb8258" />


<img width="252" height="38" alt="image" src="https://github.com/user-attachments/assets/2d4623de-d002-4ed9-a4b8-5954f2952949" />




## Training Loss Vs Iteration Plot :

<img width="800" height="579" alt="image" src="https://github.com/user-attachments/assets/fdc66eac-24a4-42b2-9564-4cb2a3d283ea" />



## New Sample Data Prediction :
<img width="332" height="47" alt="image" src="https://github.com/user-attachments/assets/c81648d7-4fca-4210-a975-99ddf730fc13" />




## RESULT : 
Thus, a neural network regression model was successfully developed and trained using PyTorch.

