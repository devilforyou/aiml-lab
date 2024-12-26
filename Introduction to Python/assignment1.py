import random
import string
import time
import uuid
import os
import psutil

def generate_password(password_length, character_set):
    """
    Generates a random password of the specified length using the given character set.

    Args:
        password_length: The desired length of the password.
        character_set: A string containing all the allowed characters for the password.

    Returns:
        The generated password as a string.
    """
    return ''.join(random.choice(character_set) for _ in range(password_length))

def main():
    """
    Main function to handle user input, generate password, and display results.
    """
    # Get desired password length from the user
    password_length = int(input("Enter the length of the password: "))

    # Display character set options to the user
    print("Choose character set:")
    print("1. Digits")
    print("2. Letters")
    print("3. Digits and Letters")
    print("4. Digits, Letters, and Punctuation")

    # Get user's character set choice
    character_set_choice = int(input("Enter your choice (1-4): "))

    # Determine the character set based on user's choice
    if character_set_choice == 1:
        character_set = string.digits
    elif character_set_choice == 2:
        character_set = string.ascii_letters
    elif character_set_choice == 3:
        character_set = string.ascii_letters + string.digits
    elif character_set_choice == 4:
        character_set = string.ascii_letters + string.digits + string.punctuation
    else:
        print("Invalid choice!")
        return

    # Start timing the password generation process
    start_time = time.time()

    # Generate the password
    generated_password = generate_password(password_length, character_set)

    # End timing the password generation process
    end_time = time.time()

    # Get the current process's ID
    current_process_id = os.getpid()

    # Get the current process object
    current_process = psutil.Process(current_process_id)

    # Get the memory usage of the current process
    memory_used = current_process.memory_info().rss / 1024  # in KB

    # Get the MAC address of the current machine
    mac_address = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])

    # Calculate the time taken to generate the password
    time_taken = end_time - start_time

    # Print the generated password, MAC address, time taken, and memory used
    print(f"Generated Password: {generated_password}")
    print(f"MAC Address: {mac_address}")
    print(f"Time Taken: {time_taken:.6f} seconds")
    print(f"Memory Used: {memory_used:.2f} KB")

if __name__ == "__main__":
    main()
