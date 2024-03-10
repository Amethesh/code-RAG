def remove_content(string1, string2):
    # Split string2 into individual lines
    lines_to_remove = string2.split("\n")

    # Remove each line from string1
    for line in lines_to_remove:
        string1 = string1.replace(line, "")

    # Remove extra whitespaces resulting from the removal
    string1 = "\n".join(line for line in string1.splitlines() if line.strip())

    return string1


# Example usage
string1 = """
import java.util.Scanner;

public class MultiplyNumbers {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);


        System.out.print("Enter the first number: ");

        double num1 = scanner.nextDouble();


        System.out.print("Enter the second number: ");

        double num2 = scanner.nextDouble();


        scanner.close();


        double result = num1 * num2;


        System.out.println("The product of " + num1 + " and " + num2 + " is: " + result);
    }
}

"""
string2 = """
double result = num1 * num2;
System.out.println("The product of " + num1 + " and " + num2 + " is: " + result);
"""

result = remove_content(string1, string2)
print("Result after removing content:")
print(result)
