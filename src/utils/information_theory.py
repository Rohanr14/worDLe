import numpy as np
from collections import Counter

def calculate_entropy(word_list):
    """Calculate the entropy of the current word list."""
    letter_counts = Counter(letter for word in word_list for letter in word)
    total_letters = sum(letter_counts.values())
    
    entropy = 0
    for count in letter_counts.values():
        p = count / total_letters
        entropy -= p * np.log2(p)
    
    return entropy

def calculate_information_gain(guess, word_list):
    """Calculate the expected information gain for a guess."""
    initial_entropy = calculate_entropy(word_list)
    
    pattern_buckets = {}
    for word in word_list:
        pattern = generate_pattern(guess, word)
        if pattern not in pattern_buckets:
            pattern_buckets[pattern] = []
        pattern_buckets[pattern].append(word)
    
    expected_entropy = 0
    for bucket in pattern_buckets.values():
        p = len(bucket) / len(word_list)
        expected_entropy += p * calculate_entropy(bucket)
    
    return initial_entropy - expected_entropy

def generate_pattern(guess, target):
    """Generate a pattern of correct, present, and absent letters."""
    pattern = ['0'] * len(guess)
    target_letters = list(target)
    
    # First pass: mark correct letters
    for i in range(len(guess)):
        if guess[i] == target[i]:
            pattern[i] = '2'
            target_letters[i] = None
    
    # Second pass: mark present letters
    for i in range(len(guess)):
        if pattern[i] == '0' and guess[i] in target_letters:
            pattern[i] = '1'
            target_letters[target_letters.index(guess[i])] = None
    
    return ''.join(pattern)