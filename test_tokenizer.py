from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
    "data/tokenizer/vocab.json",
    "data/tokenizer/merges.txt",
)

sample = "Patient was diagnosed with hypertension and diabetes."
encoded = tokenizer.encode(sample)
print("Tokens:", encoded.tokens)
print("Decoded:", tokenizer.decode(encoded.ids))
