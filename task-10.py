import numpy as np

vocab = ["I", "like", "machine", "translation"]

def softmax(logits):
    exp_vals = np.exp(logits - np.max(logits))
    return exp_vals / np.sum(exp_vals)

sentence_greedy = []
sentence_sampling = []

np.random.seed(1)

for step in range(3):
    logits = np.random.randn(len(vocab))
    probs = softmax(logits)

    greedy_index = np.argmax(probs)
    greedy_word = vocab[greedy_index]
    sentence_greedy.append(greedy_word)

    sampled_index = np.random.choice(len(vocab), p=probs)
    sampled_word = vocab[sampled_index]
    sentence_sampling.append(sampled_word)

    print(f"Step {step+1}")
    print("Logits:", logits)
    print("Probabilities:", probs)
    print("Greedy Choice:", greedy_word)
    print("Sampled Choice:", sampled_word)
    print()

print("Greedy Sentence:", " ".join(sentence_greedy))
print("Sampled Sentence:", " ".join(sentence_sampling))


