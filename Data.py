import torch
import torch.nn.functional as F
import torchtext
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from torchtext.vocab import build_vocab_from_iterator


def vocab(sentences: list) -> torchtext.vocab.Vocab:
    tokenized_sentence = []
    for sentence in sentences:
        words = sentence.split()
        tokenized_sentence.append(words)
    pad_token = "<pad>"
    unk_token = "<unk>"
    vocab = build_vocab_from_iterator(tokenized_sentence, specials=[pad_token, unk_token])
    vocab.set_default_index(vocab[unk_token])
    return vocab


def Remove_StopWord(train, val, test):
    data_trainClean = []
    data_valClean = []
    data_testClean = []

    for line in train:
        data_trainClean.append(remove_stopwords(line))
    for line in val:
        data_valClean.append(remove_stopwords(line))
    for line in test:
        data_testClean.append(remove_stopwords(line))
    return data_trainClean, data_valClean, data_testClean


def encode_data(data, vocab):
    dataN = [encode_phrase_one_hot(phrase, vocab) for phrase in data]
    return dataN


def encode_phrase_one_hot(phrase, vocabulaire):
    vocab_size = len(vocabulaire)
    ids = [vocabulaire[word] for word in phrase]
    one_hot_vectors = [F.one_hot(torch.tensor(id_mot), num_classes=vocab_size).float() for id_mot in ids]
    return one_hot_vectors


def encode_sentence_to_one_hot(sentence: str, vocab: torchtext.vocab.Vocab):
    one_hot_tensor = torch.zeros(len(sentence.split()), len(vocab) + 1)

    for i, word in enumerate(sentence.split()):
        word_id = vocab[word]
        one_hot_tensor[i] = F.one_hot(torch.tensor(word_id), num_classes=len(vocab) + 1)

    return one_hot_tensor


def TF_IDF(data, threshold=0.1):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    # filtered_sentences = [' '.join(word for word in sentence if word != '<PAD>') for sentence in line]

    tfidf_scores = {word: X[:, idx].mean() for word, idx in vectorizer.vocabulary_.items()}
    filtered_words = {word for word, score in tfidf_scores.items() if score >= threshold}
    print("filtered_words : ", filtered_words)
    return filtered_words


# def encode_data_TFIDF(data):
#     dataN = [calculatetfidf(phrase) for phrase in data]
#     return dataN


def PAD_Words(words):
    if len(words) < 20:
        words += ['<PAD>'] * (20 - len(words))  # Ajout des tokens <PAD>
    if len(words) > 20:
        words = words[:20]
    return words


def load_file(file):
    texts = []
    sentiments = []

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')

            if len(parts) == 2:
                text, sentiment = parts
                texts.append(text)
                sentiments.append(sentiment)

    return texts, sentiments


def create_global_vocab(train_data, validation_data, test_data):
    all_data = train_data + validation_data + test_data
    global_vocab = set()

    for sentence in all_data:
        for word in sentence:
            global_vocab.add(word)

    return global_vocab
