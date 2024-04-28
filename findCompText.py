import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)


r'''
#Train Model

# Import data
data = pd.read_csv(r"C:\Users\Janak\Life\AI\scienceProject2023\Medical terms dataset - Sheet1.csv", engine="python")
data.head()

# Label counts
data['term'].value_counts()

data = data[['term', 'label']]
data = data[0:9560]
data.head()

# Check for missing values in y
data.dropna(subset=['label'], inplace=True)
# Now, X and y do not contain rows with missing target values
# Train-test split
X = list(data["term"])
y = list(data["label"])
#split data in validation and training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
y_train = [int(label) for label in y_train]  # Convert labels to integers
y_val = [int(label) for label in y_val]  # Convert labels to integers

# Ensure X_train and X_val are lists of strings
X_train = [str(term) for term in X_train]
X_val = [str(term) for term in X_val]

# Shorter than will be padded, longer will be truncated
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# Define compute_metrics function
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    num_train_epochs=1,
    per_device_train_batch_size=16
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

trainer.save_model('CustomModel_v02.model')

'''

model_2 = BertForSequenceClassification.from_pretrained("CustomModel_v02.model")

userInput = input("Enter a word ")
np.set_printoptions(suppress=True)
text = userInput
inputs = tokenizer(text,padding = True, truncation = True, return_tensors='pt')
outputs = model_2(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.cpu().detach().numpy()

for predictOne in predictions:
    pred = predictOne[0]
    if pred < 0.5:
        print(userInput + "is most likely a scientific word!")

    else:
        print(userInput + "is most likely not a scientific word!")
    break

'''
predictions2 = [predictions]
print(predictions2)
if predictions2> 0.5:
    print("Regular Word")
else:
    print("complex medical word")
print(predictions)
'''