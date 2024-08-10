import random
import time
import psutil
import uuid
import sys
def print_board(board):
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("---------")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("---------")
    print(f"{board[6]} | {board[7]} | {board[8]}")

def check_win(board):
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != " ":
            return True
    return False

def computer_move(board):
    # Check if computer can win in the next move
    for i in range(len(board)):
        if board[i] == " ":
            board[i] = "O"
            if check_win(board):
                return
            board[i] = " "
    
    # Check if user can win in the next move, block them
    for i in range(len(board)):
        if board[i] == " ":
            board[i] = "X"
            if check_win(board):
                board[i] = "O"
                return
            board[i] = " "
    
    # Otherwise, make a random move
    available_moves = [i for i, x in enumerate(board) if x == " "]
    if available_moves:
        move = random.choice(available_moves)
        board[move] = "O"
    else:
        print("It's a draw!")
        return 999


def user_move(board):
    move = int(input("Enter your move (1-9): ")) - 1
    if board[move] == " ":
        board[move] = "X"
    else:
        print("Invalid move, try again.")
        user_move(board)
    

def play_game():
    board = [" "] * 9
    while True:
        print_board(board)
        user_move(board)
        if check_win(board):
            print_board(board)
            print("User wins!")
            break
        p=computer_move(board)
        if check_win(board) and p!=999:
            print_board(board)
            print("Computer wins!")
            break
        if p==999:
            break
mac = uuid.getnode()
start_time = time.time()
memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
play_game()
memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
end_time = time.time()
mac_address = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
print(f"MAC Address: {mac_address}")
print(f"Memory used: {memory_after - memory_before:.2f} MB")
print(f"Time taken: {end_time - start_time:.4f} seconds")

