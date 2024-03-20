import random

class Player():
    def __init__(self, letter):
        self.letter = letter

    def choose_column(self, game):
        pass

class RandomComputerPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def choose_column(self, game):
        return random.choice(game.available_columns())

class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def choose_column(self, game):
        column = int(input("Choose column: "))
        while column not in game.available_columns():
            column = int(input(f"Choose free column {game.available_columns()}: "))
        return column
