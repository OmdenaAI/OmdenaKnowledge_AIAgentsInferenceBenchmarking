# filename: factorial.py
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

num = int(input("Enter a positive integer: "))
result = factorial(num)
print(f"The factorial of {num} is {result}.")