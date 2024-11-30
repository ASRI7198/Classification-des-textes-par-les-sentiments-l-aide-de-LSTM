import Data
from LSTM import LSTM
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# <----------------------------- Préparation des données ------------------------------>

file_train = "DataSet/train.txt"
file_test = "DataSet/test.txt"
file_val = "DataSet/val.txt"

texts_train, sentiments_train = Data.load_file(file_train)
texts_test, sentiments_test = Data.load_file(file_test)
texts_val, sentiments_val = Data.load_file(file_val)

vocabulaire = Data.vocab(texts_train)

texts_train, texts_valid, texts_test = Data.Remove_StopWord(texts_train, texts_val, texts_test)

texts_train = [phrase.split() for phrase in texts_train]
padded_texts_train = [Data.PAD_Words(Words) for Words in texts_train]
padded_texts_train = [' '.join(phrase) for phrase in padded_texts_train]

texts_test = [phrase.split() for phrase in texts_test]
padded_texts_test = [Data.PAD_Words(Words) for Words in texts_test]
padded_texts_test = [' '.join(phrase) for phrase in padded_texts_test]

texts_valid = [phrase.split() for phrase in texts_valid]
padded_texts_valid = [Data.PAD_Words(Words) for Words in texts_valid]
padded_texts_valid = [' '.join(phrase) for phrase in padded_texts_valid]
#
# TFIDF_train = Data.TF_IDF(padded_texts_train)
# print("TFIDF_train : ", TFIDF_train)
# TFIDF_test = Data.TF_IDF(padded_texts_test)
# TFIDF_valid = Data.TF_IDF(padded_texts_valid)
#
sentiment_to_idx = {sentiment: idx for idx, sentiment in enumerate(set(sentiments_train))}
labels_train = [sentiment_to_idx[sentiment] for sentiment in sentiments_train]
labels_val = [sentiment_to_idx[sentiment] for sentiment in sentiments_val]
labels_test = [sentiment_to_idx[sentiment] for sentiment in sentiments_test]


class DatasetN(Dataset):
    def __init__(self, phrases, sentiments,vocabulaire):
        self.vocabulaire = vocabulaire
        self.phrases = phrases
        self.sentiments = torch.tensor(sentiments, dtype=torch.float32)

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        encodings_one_hot = Data.encode_sentence_to_one_hot(phrase, self.vocabulaire)
        sentiment = self.sentiments[idx]
        return encodings_one_hot, sentiment


train_dataset = DatasetN(padded_texts_train, labels_train,vocabulaire)
val_dataset = DatasetN(padded_texts_valid, labels_val,vocabulaire)
test_dataset = DatasetN(padded_texts_test, labels_test,vocabulaire)

# class DatasetN(Dataset):
#     def __init__(self, phrases, sentiments, vocabulaire):
#         self.vocabulaire = vocabulaire
#         self.phrases = phrases
#         self.sentiments = torch.tensor(sentiments, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.phrases)
#
#     def __getitem__(self, idx):
#         phrase = self.phrases[idx]
#         encodings_one_hot = Data.encode_sentence_to_one_hot(phrase, self.vocabulaire)
#         sentiment = self.sentiments[idx]
#         return encodings_one_hot, sentiment
#
#
# train_dataset = DatasetN(TFIDF_train, labels_train, vocabulaire)
# val_dataset = DatasetN(TFIDF_valid, labels_val, vocabulaire)
# test_dataset = DatasetN(TFIDF_test, labels_test, vocabulaire)

Data_loader_train = DataLoader(train_dataset, batch_size=8)
Data_loader_val = DataLoader(val_dataset, batch_size=8)
Data_loader_test = DataLoader(test_dataset, batch_size=8)

# <----------------------------- Apprentissage du RNN ------------------------------>

vocab_size = len(vocabulaire)
model = LSTM(input_size=vocab_size + 1, emb_size=64, hidden_size=64, output_size=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

best_val_loss = float('inf')
p = 20
t = 0

# <------------------------------------ Train -------------------------------------->


for epoch in range(50):
    loss_total_train = 0
    for phrase, label in Data_loader_train:
        hidden , cell = model.initHidden(batch_size=8)
        optimizer.zero_grad()
        for word in phrase.permute(1, 0, 2):
            output_train, hidden, cell = model(word, (hidden, cell))

        loss = criterion(output_train, label.to(torch.int64))
        loss_total_train += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = loss_total_train / len(Data_loader_train)

    model.eval()
    total_loss_valid = 0
    with torch.no_grad():
        for phrase, label in Data_loader_val:
            hidden_val ,cell_val = model.initHidden(batch_size=8)
            for word in phrase.permute(1, 0, 2):
                output_val, hidden_val,cell_val = model(word, (hidden_val, cell_val))
            # print(f"output : {output_val} et label : {label}")
            loss_val = criterion(output_val, label.to(torch.int64))
            total_loss_valid += loss_val.item()

    val_loss = total_loss_valid / len(Data_loader_val)
    print(f"Epoch {epoch}, Train loss: {train_loss} , Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        t = 0
    else:
        t += 1

    if t >= p:
        print("Early stopping!")
        break

torch.save(model.state_dict(), 'poids_model_LSTM.pth')

# #------------------------------------ Test -----------------------------------------------

model.eval()
acc = 0
totalA = 0
with torch.no_grad():
    for phrase, label in Data_loader_val:
        hidden_test,cell_test = model.initHidden(batch_size=8)
        phrase_loss_test = 0
        for word in phrase.permute(1, 0, 2):
            output_test, hidden_test,cell_test = model(word, (hidden_test,cell_test))
        acc += (torch.argmax(output_test, dim=1) == label).sum().item()
        totalA += 1

ACC = acc / totalA
print(f" ACC Data : {ACC:.4f}")
