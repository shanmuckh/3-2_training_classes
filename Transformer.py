import os
import numpy as np
import tensorflow as tf
import kagglehub

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Concatenate, Embedding
from tensorflow.keras.models import Model

path = kagglehub.dataset_download("roblexnana/the-babi-tasks-for-nlp-qa-system")

def find_file(base_path, filename):
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(filename)

train_file = find_file(path, "qa1_single-supporting-fact_train.txt")
test_file = find_file(path, "qa1_single-supporting-fact_test.txt")

def parse_babi(file_path):
    stories, questions, answers = [], [], []
    story = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            nid, text = line.split(" ", 1)
            if nid == "1":
                story = []
            if "\t" in text:
                q, a, _ = text.split("\t")
                stories.append(" ".join(story))
                questions.append(q)
                answers.append(a)
            else:
                story.append(text)
    return stories, questions, answers

train_stories, train_questions, train_answers = parse_babi(train_file)
test_stories, test_questions, test_answers = parse_babi(test_file)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_stories + train_questions)

vocab_size = len(tokenizer.word_index) + 1
max_story_len = max(len(s.split()) for s in train_stories)
max_question_len = max(len(q.split()) for q in train_questions)

def vectorize(stories, questions, answers):
    s = pad_sequences(tokenizer.texts_to_sequences(stories), maxlen=max_story_len)
    q = pad_sequences(tokenizer.texts_to_sequences(questions), maxlen=max_question_len)
    a = np.array([tokenizer.word_index[x] for x in answers])
    return s, q, a

x_story, x_question, y = vectorize(train_stories, train_questions, train_answers)
x_story_test, x_question_test, y_test = vectorize(test_stories, test_questions, test_answers)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(max_len, embed_dim)

    def call(self, x):
        length = tf.shape(x)[1]
        positions = tf.range(0, length)
        positions = self.pos_emb(positions)
        positions = tf.expand_dims(positions, 0)
        x = self.token_emb(x)
        return x + positions

def transformer_encoder(x, embed_dim, num_heads, ff_dim):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn)
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(embed_dim)(ffn)
    return LayerNormalization(epsilon=1e-6)(x + ffn)

embed_dim = 64

story_input = Input(shape=(max_story_len,))
question_input = Input(shape=(max_question_len,))

story_embed = PositionalEmbedding(max_story_len, vocab_size, embed_dim)(story_input)
question_embed = PositionalEmbedding(max_question_len, vocab_size, embed_dim)(question_input)

story_encoded = transformer_encoder(story_embed, embed_dim, 2, 128)
question_encoded = transformer_encoder(question_embed, embed_dim, 2, 128)

qa_attention = MultiHeadAttention(num_heads=2, key_dim=embed_dim // 2)(
    query=question_encoded,
    value=story_encoded,
    key=story_encoded
)

story_vec = GlobalAveragePooling1D()(qa_attention)
question_vec = GlobalAveragePooling1D()(question_encoded)

merged = Concatenate()([story_vec, question_vec])
output = Dense(vocab_size, activation="softmax")(merged)

model = Model([story_input, question_input], output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    [x_story, x_question],
    y,
    batch_size=32,
    epochs=30,
    validation_split=0.1
)

loss, acc = model.evaluate([x_story_test, x_question_test], y_test)
print("Final Transformer QA Accuracy:", acc)
