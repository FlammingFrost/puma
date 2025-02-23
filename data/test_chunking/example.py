def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

class Calculator:
    def __init__(self):
        pass

    def divide(self, a, b):
        return a / b if b != 0 else None
    
    def add(a, b):
        return a + b

    def subtract(a, b):
        return a - b

    def multiply(a, b):
        return a * b

    def divide_2(a, b):
        if b == 0:
            return "Cannot divide by zero"
        return a / b
    
class StringManipulation:
    def __init__(self):
        pass
    
    def reverse_string(s):
        return s[::-1]

    def count_vowels(s):
        return sum(1 for char in s.lower() if char in "aeiou")

    def is_palindrome(s):
        return s == s[::-1]

class ListManipulation:
    def __init__(self):
        pass
    
    def find_max(lst):
        return max(lst) if lst else None

    def find_min(lst):
        return min(lst) if lst else None

    def remove_duplicates(lst):
        return list(set(lst))
    
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)
    
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return f"Deposited {amount}. New balance: {self.balance}"

    def withdraw(self, amount):
        if amount > self.balance:
            return "Insufficient funds"
        self.balance -= amount
        return f"Withdrew {amount}. New balance: {self.balance}"


