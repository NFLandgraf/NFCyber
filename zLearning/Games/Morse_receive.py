import time
start_time = time.time()

letter_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                     "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
                     "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", ",", "?", "!", " "]
letter_morse_list = ["01", "1000", "1010", "100", "0", "0010", "110", "0000", "00", "0111", "101", "0100", "11", "10", "111", "0110", "1101", "010", "0000", "1", "001", "0001", "011", "1001", "1011", "1100",
                     "01", "1000", "1010", "100", "0", "0010", "110", "0000", "00", "0111", "101", "0100", "11", "10", "111", "0110", "1101", "010", "0000", "1", "001", "0001", "011", "1001", "1011", "1100"
                     "01111", "00111", "00011", "00001", "00000", "10000", "11000", "11100", "11110", "11111", "010101", "110011", "001100", "101011", "000000"]
LED_off = True
LED_on = True
signal_light = True
signal_dark = True

transmitted_morse = []
morse_continues = True

def signal_length():
    while signal_dark:
        LED_off
    time1 = time.time()
    print("Light turned on")
    while signal_light:
        LED_on
    time2 = time.time()
    print("Light turned off")
    time_diff = time2 - time1
    print("Duration of light_on: " + str(time_diff))
    return time_diff

#return True if initiation worked
def check_initiation():
    correct_initiation = [0, 1, 2]
    initiation = []
    check = True
    print("Searching for initiation sequence...")
    while check:
        time_diff = signal_length()
        if initiation == correct_initiation:
            check = False
        elif 4.9 <= time_diff <= 5.1:
            initiation.append(0)
            print("Initiation sequence 1/3")
        elif 0.9 <= time_diff <= 1.1:
            initiation.append(1)
            print("Initiation sequence 2/3")
        elif 2.9 <= time_diff <= 3.1:
            initiation.append(2)
            print("Initiation sequence 3/3")
        else:
            initiation = []
    print("Initiation successful")

#ergänzt transmitted_morse mit "morse_letter", stoppt bei 2s LED_on
def transmit_letter():
    global transmitted_morse
    global morse_continues
    morse_letter = []
    letter_continues = True
    while letter_continues:
        time_diff = signal_length()
        if 4.9 <= time_diff <= 5.1:
            letter_continues = False
        elif 9.9 <= time_diff <= 10.1:
            morse_continues = False
            letter_continues = False
        elif 0.4 <= time_diff <= 0.6:
            morse_letter.append(0)
        elif 0.9 <= time_diff <= 1.1:
            morse_letter.append(1)
    for i in range(len(morse_letter)):
        morse_letter[i] = str(morse_letter[i])
    morse_letter_str = "".join(morse_letter)

    print("New Morseletter added to transmitted Morse: " + morse_letter_str)
    transmitted_morse.append(morse_letter_str)

def transmit_final():
    global transmitted_morse
    while morse_continues:
        transmit_letter()
    print("Transmitted Morse: " + str(transmitted_morse))

def compute():
    final_sentence = []
    check = True
    i = 0
    for letter in range(len(transmitted_morse)):
        while check:
            if i == len(letter_morse_list):
                i = 0
            elif transmitted_morse[letter] == letter_morse_list[i]:
                final_sentence.append(letter_list[i])
                i = 0
                check = False
            elif transmitted_morse[letter] != letter_morse_list[i]:
                i = i + 1
        print("Translation for morse_letter found")
    print("Translation finished! Sentence: " + final_sentence)
    final = "".join(final_sentence)
    print(final)

check_initiation()
transmit_final()
compute()

end_time = time.time()
print("\nDone (" + str(end_time - start_time) + "s)")