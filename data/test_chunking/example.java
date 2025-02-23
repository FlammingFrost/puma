import java.util.Scanner;

public class Example {
    public static int add(int a, int b) {
        return a + b;
    }

    public static int multiply(int x, int y) {
        return x * y;
    }
}

public class FactorialCalculator {
    // Recursive method to calculate factorial
    public static long factorialRecursive(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }
        return n * factorialRecursive(n - 1);
    }

    // Iterative method to calculate factorial
    public static long factorialIterative(int n) {
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        scanner.close();
        
        // Calculate factorial using both methods
        long resultRecursive = factorialRecursive(number);
        long resultIterative = factorialIterative(number);
        
        // Print results
        System.out.println("Factorial (Recursive) of " + number + " is: " + resultRecursive);
        System.out.println("Factorial (Iterative) of " + number + " is: " + resultIterative);
    }
}