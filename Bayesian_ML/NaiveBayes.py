#classify whether a sentence is about Sports or Politics.

# 0 = Sports
# 1 = Politics

X_train = [
    "match game win",       # Sports
    "game sport match",     # Sports
    "win election vote",    # Politics
    "vote politics election",# Politics
    "election win vote"     # Politics
]

y_train = [0, 0, 1, 1, 1]

# The Mystery Sentence to Classify
X_test = "win vote game"

training_data = []

for sentence, label in zip(X_train, y_train):
    training_data.append([sentence.split(), label])

# 1. Initialize Universal Buckets
class_counts = {}   # Holds word counts per class
class_totals = {}   # Holds total word count per class
priors_count = {}   # Holds number of sentences per class (for P(y))

# 2. Loop through data
for words, label in training_data:
    
    # A. Setup the Class Bucket if we haven't seen this label before
    if label not in class_counts:
        class_counts[label] = {}
        class_totals[label] = 0
        priors_count[label] = 0
    
    # B. Count the sentence for the Prior P(y)
    priors_count[label] += 1
    
    # C. Count the words for Likelihood P(x|y)
    for word in words:
        class_counts[label][word] = class_counts[label].get(word, 0) + 1
        class_totals[label] += 1

print("Word Counts:", class_counts)
print("Total Words per Class:", class_totals)
print("Sentences per Class:", priors_count)

def predict(test_sentence, class_counts, class_totals, priors_count):
    # 1. Get Vocabulary Size (Unique words across all classes)
    # We need this for the smoothing math
    vocab = set()
    for counts in class_counts.values():
        vocab.update(counts.keys())
    vocab_size = len(vocab)
    
    scores = {}
    total_docs = sum(priors_count.values())

    # 2. Check every class (0, 1, etc.)
    for label in class_counts:
        
        # A. Calculate Prior P(y)
        prior = priors_count[label] / total_docs
        
        # B. Calculate Likelihood P(words | y)
        likelihood = 1.0
        words = test_sentence.split()
        
        for word in words:
            # How often does this word appear in this class?
            count = class_counts[label].get(word, 0)
            
            # MATH: (Count + 1) / (Total Words in Class + Unique Words)
            prob = (count + 1) / (class_totals[label] + vocab_size)
            
            likelihood *= prob
            
        # C. Final Score = Prior * Likelihood
        scores[label] = prior * likelihood
        
    return scores

# --- RUN IT ---
results = predict(X_test, class_counts, class_totals, priors_count)

print("Scores:", results)
# Find the winner
predict_class = max(results, key=results.get)
print(f"Prediction: Class {predict_class}")

