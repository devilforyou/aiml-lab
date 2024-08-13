import time
import psutil
import uuid
def check_sentence(sentence):
    """Checks a sentence against the given rules and returns the result."""

    rules = [
        lambda s: s[0].isupper(),  # Starts with a capital letter
        lambda s: s[-1] in ['.', '?', '!'],  # Ends with proper punctuation
        lambda s: '  ' not in s,  # No consecutive spaces
        lambda s: not any(char.isdigit() for char in s),  # No digits
        lambda s: len(s.split()) == s.count(' ') + 1  # Words separated by single space
    ]

    for i, rule in enumerate(rules):
        if not rule(sentence):
            return f"The sentence is incorrect: Rule {i+1} violated."

    return "The sentence is correct."

if __name__ == "__main__":
    sentence = input("Enter Sentences")
    mac = uuid.getnode()
    start_time = time.time()
    memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

    result = check_sentence(sentence)
    print(f"Input: {sentence}")
    print(f"Output: {result}")
    print()

    memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
    end_time = time.time()
    mac_address = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
    print(f"MAC Address: {mac_address}")
    print(f"Memory used: {memory_after - memory_before:.2f} MB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
