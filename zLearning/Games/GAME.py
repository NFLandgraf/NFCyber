from PLAYER import HumanPlayer, RandomComputerPlayer



class TicTacToe:
    def __init__(self):
        self.board = [" " for i in range(9)]
        self.current_winner = None

    @staticmethod
    def show_initial_board():
        for row in [[0, 1, 2], [3, 4, 5], [6, 7, 8]]:
            print(row)

    def print_board(self):
        updated_board = [self.board[i * 3:(i * 3) + 3] for i in range(3)]
        for row in updated_board:
            print("| " + " | ".join(row) + " |")

    def available_moves(self):
        available_moves = []
        for (spot, i) in enumerate(self.board):
            if i == " ":
                available_moves.append(spot)
        return available_moves

    def make_move(self, square, letter):
        self.board[square] = letter

    def winner(self, letter):
        # at first the rows:
        for row in [self.board[i * 3:(i * 3) + 3] for i in range(3)]:
            if all([spot == letter for spot in row]):
                return True

        # now the columns:
        updated_board = []
        for a in range(3):
            yeet = []
            for i in range(9):
                if i % 3 == 0:
                    yeet.append(self.board[i + a])
            updated_board.append(yeet)
        for column in updated_board:
            if all([spot == letter for spot in column]):
                return True

        # now the diagonals:
        runter = [self.board[i] for i in [0, 4, 8]]
        if all(spot == letter for spot in runter):
            return True
        hoch = [self.board[i] for i in [2, 4, 6]]
        if all(spot == letter for spot in hoch):
            return True
        return False


def play(game, x_player, o_player, print_board=True):
    letter = "X"
    check = True
    if print_board:
        game.show_initial_board()
    while check:
        if letter == "X":
            square = x_player.get_move(game)
            game.make_move(square, letter)
        elif letter == "O":
            square = o_player.get_move(game)
            game.make_move(square, letter)
        if print_board == True:
            game.print_board()
            print("\n__________________\n")
        if game.winner(letter):
            game.current_winner = letter
            check = False
        if game.current_winner == "X":
            print("X won!")
        elif game.current_winner == "O":
            print("O won!")
        elif " " not in game.board:
            print("Tie")

        if letter == "X":
            letter = "O"
        else:
            letter = "X"

x_player = HumanPlayer("X")
o_player = RandomComputerPlayer("O")
play(TicTacToe(), x_player, o_player, print_board=True)