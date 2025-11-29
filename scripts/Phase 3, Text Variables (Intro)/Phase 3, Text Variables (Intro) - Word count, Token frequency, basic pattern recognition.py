# Text Variables (Intro) - Word count, Token frequency, and Basic Pattern Recognition
import pandas as pd

'''
# word counts
df = pd.DataFrame({
    'Notes': [
        "Patient shows improvent in tumor size",
        "Follow-up required",
        "No significant change observed",
        "Needs urgent biopsy"
    ]
})
df['word_count'] = df['Notes'].str.split().str.len()
    # .str.split() -> splits each string into list of words with whitespace as separator
    # .str.len() -> counts no. of elements in each resulting list
print(df)

# Token frequency
from collections import Counter
all_words = " ".join(df['Notes']).lower().split()
word_frequency = Counter(all_words)
print(f"Top 5 most common words: {word_frequency.most_common(5)}")

# Basic Pattern Recognition
    # flag if 'urgent' appears in Notes
df['urgent_flag'] = df['Notes'].str.contains('urgent', case=False).astype(int)
    # case=False -> case insensitive, matches 'urgent', 'URGENT', 'Urgent', 'uRgEnT', etc
print(df)
'''

with open("D:\EDA\data\Phase 3, Text Variables (Intro)\doctors_note.txt", mode="rt", encoding='utf-8') as f:
    doctors_note = f.read() 
        # .readlines() makes list of lines
        # .read() makes a single string of all contents of the .txt file
tokens = doctors_note.lower().split()
print(f"Word count: {len(tokens)}")

critical_words = ['urgent', 'biopsy', 'counseling']
for word in critical_words:
    print(f"Is {word} present in doctor's note? {word in doctors_note} ")

from collections import Counter
token_frequency = Counter(tokens)
print(token_frequency.most_common(10))
