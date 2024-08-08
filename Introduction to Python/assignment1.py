import random
import string
import time
import uuid
import os
import psutil

def generate_password(length, charset):
    return ''.join(random.choice(charset) for _ in range(length))

def main():
    # Take user input for password length
    length = int(input("Enter the length of the password: "))

    # Provide options for character set
    print("Choose character set:")
    print("1. Digits")
    print("2. Letters")
    print("3. Digits and Letters")
    print("4. Digits, Letters, and Punctuation")
    choice = int(input("Enter your choice (1-4): "))

    if choice == 1:
        charset = string.digits
    elif choice == 2:
        charset = string.ascii_letters
    elif choice == 3:
        charset = string.ascii_letters + string.digits
    elif choice == 4:
        charset = string.ascii_letters + string.digits + string.punctuation
    else:
        print("Invalid choice!")
        return

    # Generate password
    start_time = time.time()
    password = generate_password(length, charset)
    end_time = time.time()

    # Get MAC address
    mac = uuid.getnode()
    mac_address = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])

    # Calculate time taken
    time_taken = end_time - start_time

    # Calculate memory used
    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / 1024  # in KB

    # Print results
    print(f"Generated Password: {password}")
    print(f"MAC Address: {mac_address}")
    print(f"Time Taken: {time_taken:.6f} seconds")
    print(f"Memory Used: {memory_used:.2f} KB")

if __name__ == "__main__":
    main()



