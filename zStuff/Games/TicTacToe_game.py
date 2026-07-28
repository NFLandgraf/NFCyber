from TicTacToe_player import HumanPlayer, RandomComputerPlayer, SmartComputerPlayer
import time
score = []

class TicTacToe:
    def __init__(self):
        self.board = [" " for i in range(9)]
        self.current_winner = None

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print("| " + " | ".join(row) + " |")

    @staticmethod
    def print_board_nums():
        # 0 | 1 | 2 tells us what number corresponds to what box
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print("| " + " | ".join(row) + " |")

    def available_moves(self):
        moves = []
        for (i, spot) in enumerate(self.board):
            if spot == " ":
                moves.append(i)
        return moves

    def empty_squares(self):
        return " " in self.board # return boolean

    def num_empty_squares(self):
        return self.board.count(" ")

    def make_move(self, square, letter):
        # if valid move, make move and assign square to letter
        # then return true, if invalid, return false
        if self.board[square] == " ":
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        #winner if 3 in a row anywhere... we have to check all of them
        # first check row
        row_ind = square // 3
        row = self.board[row_ind*3 : (row_ind + 1)*3]
        if all([spot == letter for spot in row]):
            return True

        #check column
        col_ind = square % 3
        column = [self.board[col_ind + i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True

        #check diagonals
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False

def play(game, x_player, o_player, print_game = True):
    global score
    # return winner of game (letter) or none for a tie
    if print_game:
        game.print_board_nums()
    letter = "X" #starting letter
    while game.empty_squares():
        if letter == "O":
            square = o_player.get_move(game)
        else:
            square = x_player.get_move(game)

        #define a function to macke a move
        if game.make_move(square, letter):
            if print_game:
                print(letter + f" to {square}")
                game.print_board()
                print("")

            if game.current_winner:
                if print_game:
                    print(letter + " WINS!")
                score.append(letter)
                return letter

            # after move, we need to alternate letters
            letter = "O" if letter == "X" else "X"
        time.sleep(0)
    score.append("T")
    if print_game:
        print("TIE")

def calculate_score():
    a = 0
    for i in range(100):
        x_player = SmartComputerPlayer("X")
        o_player = SmartComputerPlayer("O")
        t = TicTacToe()
        play(t, x_player, o_player, print_game=False)
        time.sleep(0)
        print(a)
        a = a + 1
    gesamt = score.count("X") + score.count("O") + score.count("T")
    x_wins = score.count("X") / gesamt * 100
    o_wins = score.count("O") / gesamt * 100
    tie = score.count("T") / gesamt * 100

    print(f"X: {x_wins}\nO: {o_wins}\nTie: {tie}")

calculate_score()

print("Done")