from VierGewinnt_player import RandomComputerPlayer, HumanPlayer

class VierGewinnt():
    def __init__(self):
        self.board = [" " for i in range(6*7)]
        self.current_winner = None

    @staticmethod
    # returns rows [0,1,2,3,4,5,6],[7,...
    def rows():
        numbers = list(range(0, 42))
        rows = [numbers[i * 7:i * 7 + 7] for i in range(6)]
        return rows

    @staticmethod
    # returns columns [0,7,14,21,28,35],[1,...
    def columns():
        b = 0
        list = []
        columns = []
        for a in range(7):
            for i in range(6):
                list.append(i * 7 + b)
            columns.append(list)
            b += 1
            list = []
        return columns

    @staticmethod
    # returns diagonals /
    def diagonals_up(game):
        # creating a list with specs, which spot of which row is needed, "X" is skipped
        specs = []
        x = 7
        for _ in range(12):
            check = []
            i = 7
            for _ in range(6):
                check.append(i - x)
                i -= 1
            x -= 1
            for q, spot in enumerate(check):
                if spot < 0 or spot > 6:
                    check[q] = "X"
            specs.append(check)

        # translate our specs list into the respective rows
        diagonal_up = []
        for dia in specs:
            check2 = []
            for i, spot in enumerate(dia):
                if spot == "X":
                    pass
                else:
                    check2.append(game.rows()[i][spot])
            diagonal_up.append(check2)
        return diagonal_up

    # returns board rows that reflect the actual board
    def board_rows(self):
        board_rows = [self.board[i*7:i*7+7] for i in range(6)]
        return board_rows

    def print_board(self):
        ind = [str(i) for i in range(7)]
        print(" " + "  ".join(ind))
        for row in self.board_rows():
            print("| " + "| ".join(row) + "|")

    def available_columns(self):
        available_columns = []
        for a, column in enumerate(self.columns()):
            for spot in column:
                if self.board[spot] == " ":
                    available_columns.append(a)
                break
        return available_columns

    def make_move(self, letter, column):
        Coi = self.columns()[column]
        l = len(Coi) - 1
        while l >= 0:
            spot = Coi[l]
            if self.board[spot] == " ":
                self.board[spot] = letter
                break
            else:
                l -= 1
        return spot

    def winner_row(self, letter, square):
        # row_oi is our row [...]
        checks = 0
        for i, row in enumerate(game.rows()):
            for a, spot in enumerate(row):
                if spot == square:
                    row_oi = row
        for i in range(len(row_oi)):
            if self.board[row_oi[i]] == letter:
                checks += 1
            else:
                checks = 0
            if checks >= 4:
                return True

    def winner_column(self, letter, square):
        # column_oi is our column [...]
        checks = 0
        for i, column in enumerate(game.columns()):
            for a, spot in enumerate(column):
                if spot == square:
                    column_oi = column
        for i in range(len(column_oi)):
            if self.board[column_oi[i]] == letter:
                checks += 1
            else:
                checks = 0
            if checks >= 4:
                return True

    def winner_diagonal_up(self, game, letter, square):
        self.diagonals_up(game)

    def winner(self, letter, square):
        if self.winner_row(letter, square):
            self.current_winner = letter
            return True
        elif self.winner_column(letter, square):
            self.current_winner = letter
            return True
        elif self.winner_diagonal_up(game, letter, square):
            self.current_winner = letter
            return True
        return False


def play(game, x_player, o_player, print_board=True):
    letter = "X"
    game.print_board()

    while " " in game.board:
        if letter == "X":
            column = x_player.choose_column(game)
            spot = game.make_move(letter, column)
        else:
            column = o_player.choose_column(game)
            spot = game.make_move(letter, column)
        if print_board:
            game.print_board()
        if game.winner(letter, spot):
            print(f"Player {letter} won!")
            break
        letter = "O" if letter == "X" else "X"




x_player = HumanPlayer("X")
o_player = RandomComputerPlayer("O")
game = VierGewinnt()
play(game, x_player, o_player, print_board=True)