import time
new_word = input("Give word: ")
print("Word: " + new_word)
letter_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                     "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
                     "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", ",", "?", "!", " "]
letter_morse_list = ["01", "1000", "1010", "100", "0", "0010", "110", "0000", "00", "0111", "101", "0100", "11", "10", "111", "0110", "1101", "010", "0000", "1", "001", "0001", "011", "1001", "1011", "1100",
                     "01", "1000", "1010", "100", "0", "0010", "110", "0000", "00", "0111", "101", "0100", "11", "10", "111", "0110", "1101", "010", "0000", "1", "001", "0001", "011", "1001", "1011", "1100"
                     "01111", "00111", "00011", "00001", "00000", "10000", "11000", "11100", "11110", "11111", "010101", "110011", "001100", "101011", "000000"]
new_morse = []
final_morse = []
light_on = True
light_off = True

def light_punkt():

light_strich = True
light_pause = True

#new_morse filled with 00101, 01111, ...
def create_morsefile(word):
    global new_morse
    word = []
    for letter in new_word:
        word.append(letter)
    let = 0
    i = 0
    while i < len(word):
        if let == len(letter_list):
            print("Mistake")
        elif word[i] == letter_list[let]:
            new_morse.append(letter_morse_list[let])
            i = i + 1
            let = 0
        elif word[i] != letter_list[let]:
            let = let + 1
    print("New morse: " + str(new_morse))

def initiation():
    print("Initiation...")
    light_on
    print("light_on (5)")
    time.sleep(5)
    light_pause
    print("light_off")
    light_on
    print("light_on (1)")
    time.sleep(1)
    light_pause
    print("light_off")
    light_on
    print("light_on (3)")
    time.sleep(3)
    light_pause

def transmit():
    i = 0
    while i < len(new_morse):
        letter = new_morse[i]
        for b in letter:
            if b == "0":
                light_punkt
                print("light_punkt")
            elif b == "1":
                light_strich
                print("light_strich")
        i = i + 1
        print("Letter transmitted: " + str(letter))
        time.sleep(2)
    print("Transmission finished")

def finishing_sequence():
    print("Finishing sequence...")
    light_on
    print("light_on (5)")
    time.sleep(5)
    light_off
    print("light_off")

create_morsefile(new_word)
#initiation()
transmit()
#finishing_sequence()

print("\nDone")
