# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

Named Entity Recognition (NER) is a sequence labeling task where the goal is to identify and classify named entities in text into predefined categories such as person, organization, location, date, and more. This project uses a BiLSTM model to learn contextual word representations and assign NER tags to each word in a sentence. The dataset used is a standard NER dataset containing three columns — Word, POS, and Tag — where each word is labeled with its corresponding BIO-style named entity tag across multiple sentences.

## DESIGN STEPS

### STEP 1: Data Loading and Preprocessing
The NER dataset is loaded and forward-filled to handle missing sentence identifiers. Words and tags are extracted and mapped to numerical indices. Sentences are grouped and encoded, then padded to a fixed length of 50 tokens. The data is split into 80% training and 20% test sets and wrapped in a custom PyTorch Dataset and DataLoader.

### STEP 2: Model Definition and Setup
A Bidirectional LSTM model is defined with an embedding layer, a BiLSTM layer, and a fully connected output layer that predicts a tag for each token. Cross-entropy loss is used as the criterion and Adam is used as the optimizer. The model is moved to GPU if available.

### STEP 3: Training and Evaluation
The model is trained for 3 epochs where each epoch computes training and validation loss. After training, the model is evaluated on the test set using a classification report. A sample sentence is passed through the model and predicted tags are compared against the actual tags word by word.


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
