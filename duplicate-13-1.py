user_input = input("Enter a sentence: ")
words = user_input.split()
seen = set()
duplicates = set()
for word in words:
    if word in seen:
        duplicates.add(word)
    else:
        seen.add(word)
if duplicates:
    print("Duplicates found:", duplicates)
else:
    print("No duplicates")