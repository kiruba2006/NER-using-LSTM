# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

Build a Named Entity Recognition (NER) model that can automatically identify and classify entities like names of people, locations, organizations, and other important terms from text. The goal is to tag each word in a sentence with its corresponding entity label.

<img width="622" height="737" alt="image" src="https://github.com/user-attachments/assets/5072a85b-0392-4de0-a188-76b5b45e6ffa" />

## DESIGN STEPS

### STEP 1
Import necessary libraries and set up the device (CPU or GPU).

### STEP 2
Load the NER dataset and fill missing values.

### STEP 3
Create word and tag dictionaries for encoding.

### STEP 4
Group words into sentences and encode them into numbers.

### STEP 5
Build a BiLSTM model for sequence tagging.

### STEP 6
Train the model using the training data.

### STEP 7
Evaluate the model performance on test data.


## PROGRAM
### Name: Kiruba R C
### Register Number:212224230125
```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=128):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        return logits


model = BiLSTMTagger(vocab_size=len(words), tagset_size=len(tags)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tags)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tags)), labels.view(-1))
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    return train_losses, val_losses
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="950" height="818" alt="image" src="https://github.com/user-attachments/assets/847e8d6c-5a4c-49db-8f37-419920570c83" />


### Sample Text Prediction

<img width="506" height="542" alt="image" src="https://github.com/user-attachments/assets/9267d6fd-93c8-4344-bafc-55e4de018a18" />


## RESULT
The BiLSTM model was successfully trained for 3 epochs to perform Named Entity Recognition on the NER dataset. The model learned to identify and classify named entities such as persons, organizations, locations, and geopolitical entities from input text sequences. The training and validation loss decreased consistently across epochs, indicating good convergence. The sample prediction output demonstrates that the model correctly assigns NER tags to most words in the test sentence with reasonable accuracy.
