from langchain_text_splitters.nltk import NLTKTextSplitter

import nltk
nltk.download('punkt_tab')


# Sample text to split
text = """This is the first sentence.
This is the second sentence.  
This is the third sentence, with a bit more to it.
And this is the fourth sentence!  It's exciting.

This is a new paragraph with a fifth sentence.
"""

# 1. Initialize the NLTKTextSplitter
#  - chunk_size: Maximum size of each chunk (in characters).  Adjust as needed.
#  - chunk_overlap: How much overlap between chunks (in characters).  Helps with context.
nltk_text_splitter = NLTKTextSplitter(
    chunk_size=5,
    chunk_overlap=1,
    language="english"
)
# 2. Split the text into chunks
chunks = nltk_text_splitter.split_text(text)

# 3. Print the chunks (or do something else with them)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")


# Example with different parameters

text2 = "This is the first sentence. This is the second sentence.  Now, let's have a very long sentence that spans multiple lines and contains a lot of information about different topics. Finally, we have a short sentence."
nltk_text_splitter2 = NLTKTextSplitter(
    chunk_size=50, chunk_overlap=0)  # Smaller chunk size
chunks2 = nltk_text_splitter2.split_text(text2)

print("--- Different parameters ---")
for i, chunk in enumerate(chunks2):
    print(f"Chunk {i+1}:\n{chunk}\n")
