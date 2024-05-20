from spellchecker import SpellChecker

spell = SpellChecker()

text = "This is a sampl text with speeling errors."

words = text.split()
misspelled = spell.unknown(words)

for word in misspelled:
    corrected_word = spell.correction(word)
    text = text.replace(word, corrected_word)

print(text)
