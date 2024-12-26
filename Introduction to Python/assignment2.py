import time
import psutil
import uuid

def check_sentence_grammar(sentence):
    """
    Checks a given sentence against a set of grammatical rules.

    Args:
      sentence: The sentence to be checked.

    Returns:
      A string indicating whether the sentence is correct or which rule was violated.
    """

    grammar_rules = [
        lambda s: s[0].isupper(),  # Starts with a capital letter
        lambda s: s[-1] in ['.', '?', '!'],  # Ends with proper punctuation
        lambda s: '  ' not in s,  # No consecutive spaces
        lambda s: not any(char.isdigit() for char in s),  # No digits
        lambda s: len(s.split()) == s.count(' ') + 1  # Words separated by single space
    ]

    for rule_index, rule_function in enumerate(grammar_rules):
        if not rule_function(sentence):
            return f"The sentence is incorrect: Rule {rule_index + 1} violated."

    return "The sentence is correct."

if __name__ == "__main__":
    user_input_sentence = input("Enter Sentence: ") 
    
    start_time = time.time()
    initial_memory_usage = psutil.Process().memory_info().rss / (1024 * 1024) 

    grammar_check_result = check_sentence_grammar(user_input_sentence)
    
    print(f"Input: {user_input_sentence}")
    print(f"Output: {grammar_check_result}")
    print()

    end_time = time.time()
    final_memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # Get the MAC address
    device_mac = uuid.getnode()
    mac_address = ':'.join(['{:02x}'.format((device_mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])

    print(f"MAC Address: {mac_address}")
    print(f"Memory used: {final_memory_usage - initial_memory_usage:.2f} MB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
