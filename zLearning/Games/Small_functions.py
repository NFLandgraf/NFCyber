
def number_is_gerade(number):
    a = float(number / 2)
    b = float(int((a)))
    if a == b:
        return True
    else:
        return False

def number_is_ungerade(number):
    a = float(number / 2)
    b = float(int((a)))
    if a == b:
        return False
    else:
        return True

def average_of_list(list):
    if len(list) != 0:
        return sum(list) / len(list)
    else:
        print("length of list = 0; no average calculation possible")

def listmax(list): ##finds maximum in list
    length = len(list) - 1
    max = list[0]
    i = 1

    while i <= length:
        if max - list[i] < 0:
            max = list[i]
            i = i + 1
        elif max - list[i] >= 0:
            i = i + 1
    return max

def random_hit(max, repeats): ## calculates probability of hitting number (0-max with x repeats)
    import random
    list = []
    for i in range(repeats):
        goal = random.randrange(1, max)
        guess = random.randrange(1, max)

        if goal == guess:
            list.append(100)
        else:
            list.append(0)

    avrg = sum(list) / len(list)
    print(avrg)

def fib(n): ##provides fibonacci until n
    a, b = 0, 1
    result = []
    while a < n:
        result.append(a)
        result.append(b)
        a = a + b
        b = a + b
    print(str(result))

def calculate_max_from_file(): #creates textfile with random numbers, calculates max number
    from Main import listmax
    import random
    list = []

    with open("Numbers", "w") as f:
        for i in range(5):
            f.write(str(random.randrange(1, 11)) + "\n")

    with open("Numbers", "r") as f:
        for line in f:
            list.append(int(line.rstrip("\n")))

    print(list)
    print(listmax(list))

def order_citylist_alphabetically():#orders citylist from file alphabetically
    city_list = []
    with open("C:\\Users\\nicol\Desktop\\unordered_cities.txt", "r") as f:
        for line in f:
            city_list.append(line.rstrip("\n"))

    sorted_list = sorted(city_list)

    with open("Cities", "w") as f:
        f.write("Unsorted Citylist:\n")
        for city in city_list:
            f.write(city + "\n")
        f.write("\nSorted Citylist:\n")

        for city in sorted_list:
            f.write(city + "\n")
    print("Done")