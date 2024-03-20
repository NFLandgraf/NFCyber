letter_list = []
answer = []
used_letters = []
versuche = 8

def create_letter_list():
    import random
    global letter_list
    number = random.randrange(1, 98)
    word_list = []
    with open("Hangman_words.txt", "r") as f:
        for line in f:
            word_list.append(line)

    word = word_list[0]
    for letter in word:
        letter_list.append(letter)
    del letter_list[len(letter_list) - 1]
    #print("Letter_list: " + str(letter_list))

def present_riddle():
    global answer
    length = len(letter_list)
    i = 0
    while i < length:
        answer.append("_ ")
        i = i + 1
    answer_str = "".join(answer)
    print(str(answer_str) + "\n")

def check():
    global answer, letter_list, versuche, used_letters
    ok = True
    guess = input("letter: ")
    used_letters.append(guess)
    i = 0
    check = True
    while i < len(letter_list) and check:
        if guess == letter_list[i]:
            answer[i] = guess + " "
            letter_list[i] = letter_list[i].upper()
            ok = False
        elif i == len(letter_list) - 1 and ok == True:
            check = False
            versuche = versuche - 1
        elif i == len(letter_list) - 1 and ok == False:
            check = False
        else:
            i = i + 1
    print("".join(answer) + "   (" + str(versuche) + " lives left)    Used: " + ",".join(used_letters) + "\n")

def check_final():
    i = 0
    ok = True
    while i < len(answer) and ok:
        if versuche == 0:
            print("---LOOSE---")
            print(" ".join(letter_list))
            ok = False
        elif answer[i] != "_ " and i == len(answer) - 1:
            print("---WIN---")
            print(" ".join(letter_list))
            ok = False
        elif answer[i] == "_ ":
            check()
            i = 0
        elif answer[i] != "_ ":
            i = i + 1

def again():
    global answer, letter_list, versuche, used_letters
    create_letter_list()
    present_riddle()
    check_final()
    while input("\nAgain? (y/n): ") == "y":
        letter_list = []
        answer = []
        used_letters = []
        versuche = 8
        create_letter_list()
        present_riddle()
        check_final()

again()

print("Done")