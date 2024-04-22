#functions for simple calculations

#Addition
def add(a,b):
    return a+b


#Subtraction
def subtract(a,b):
    return a-b

#Multiplication
def multiply(a,b):
    return a*b

#Division
def division (a,b):
    return a/b

#Square
def square(a):
    return a*a

#Square root
def squareroot(a):
    return a**0.5

#User prompt

welcome="""Welcome to my calculator!
        1. Addition
        2. Subtraction
        3. Multiplication
        4. Divide
        5. Square
        6. Square root
        Please select an option: 1, 2, 3, 4, 5, 6 for the corresponding function"""

print (welcome)

#Programm loop 
check="y"

while check=="y":
    #User input to select math function
    sel = int(input("What function would you like to use?"))

    #User input for numbers

    number_1 = int(input("Your first number to calculate with:"))
    print(number_1)
    number_2 = int(input("Your second number to calculate with:"))
    print(number_2)
    #Evaluating user input and preventing exceptions

    if sel == 1:
        print("You have selected addition")
        print(number_1,"+",number_2,"=",
            add(number_1,number_2))
    
    elif sel == 2:
        print("You have selected subtraction")
        print(number_1,"-",number_2,"=",
            subtract(number_1,number_2))
    
    elif sel == 3:
        print("You have selected multiplication")
        print(number_1,"*",number_2,"=",
            multiply(number_1,number_2))
    
    elif sel == 4:
        print("You have selected division")
        print(number_1,"/",number_2,"=",
            division(number_1,number_2))
    
    elif sel == 5:
        print("You have selected squaring")
        print("The squre of",number_1,"is",square(number_1))

    elif sel == 6:
        print("You have selected square root")
        print("The square root of",number_1,"is",squareroot(number_1))

    else:
        print("Invalid input, please select a number between 1-6 corresponding to the listed functions.")


    #End of while loop
    check = input("Would you like to continue? y/n")
    if check=="y":
        print(welcome)
        continue

    if check =="n":
        print("Thank you for using my calculator!")
        break
else:
    print("please type y or n to continue or exit the program.")