# 34-Python-Code-
A collection of 34 Python Programs covering basic to intermediate concepts.

This repository contains a collection of python functions and implementing various concepts from number theory, discrete mathematics, and computational algorithms. Each function is designed to be efficient, and the included source code often includes performance metrics (execution time and memory usage) for benchmarking purposes. 


# Project Highlights

1. Diverse mathematical function: from baisc arithmetic properties(factorial, mean of digits) to advanced number theory (modular arithmetic, carmichael numbers, riemann zeta approximation).
2. Performance measurement: most code blocks include utility code to measure execution time and memory usage (using time and sys/tracemalloc).
3. Fundamental algorithms: implementation of core algorithms like modular exponentiation, extended euclidean algorithm, and pollard's rho for factorization.

# Function
1. Factorial Calculation: Calculates $n!$ for non-negative integers using an iterative approach.

2. Palindrome Check: Determines if a number reads the same forwards and backwards using string slicing.

3. Mean of Digits: Computes the arithmetic average of all digits in a given number.

4. Digital Root: Recursively sums the digits of a number until a single-digit result is obtained.

5. Abundant Number Check: Checks if the sum of proper divisors of $n$ is greater than $n$.

6. Deficient Number Check: Checks if the sum of proper divisors of $n$ is less than $n$.

7. Harshad Number: Determines if a number is divisible by the sum of its digits.

8. Automorphic Number: Checks if the square of a number ends with the number itself (e.g., $5^2 = 25$).

9. Pronic Number: Verifies if a number is the product of two consecutive integers ($n(n+1)$).

10. Prime Factors: Generates a list of all prime factors for a given number using optimized trial division.

11. Count Distinct Prime Factors: Returns the count of unique prime factors using Set data structures.

12. Prime Power Check: Checks if a number can be expressed as $p^k$ (where $p$ is prime).

13. Mersenne Prime Check: Uses the Lucas-Lehmer test logic to check primality of $2^p - 1$.

14. Twin Primes Generator: Generates all twin prime pairs $(p, p+2)$ up to a specified limit.

15. Count of Divisors $d(n)$: Calculates the total number of divisors efficiently in $O(\sqrt{n})$.

16. Aliquot Sum: Calculates the sum of proper divisors (used to identify Perfect Numbers).

17. Amicable Numbers: Checks if two numbers form a pair where the aliquot sum of one equals the other.

18. Multiplicative Persistence: Counts the steps required to reduce a number to a single digit by multiplying its digits.

19. Highly Composite Numbers: Checks if a number has strictly more divisors than any smaller positive integer.

20. Modular Exponentiation: Efficiently computes $(base^{exp}) \% mod$ using the square-and-multiply algorithm.

21. Modular Multiplicative Inverse: Finds $x$ such that $ax \equiv 1 \pmod m$ using the Extended Euclidean Algorithm.

22. Chinese Remainder Theorem (CRT): Solves a system of simultaneous linear congruences.

23. Quadratic Residue Check: Uses Euler's Criterion to determine if $x^2 \equiv a \pmod p$ has a solution.

24. Multiplicative Order: Finds the smallest $k$ such that $a^k \equiv 1 \pmod n$.

25. Fibonacci Prime Check: Determines if a number is both a Fibonacci number and a Prime number.

26. Lucas Numbers: Generates the first $n$ terms of the Lucas sequence ($2, 1, 3, 4, 7...$).

27. Perfect Powers: Checks if a number can be expressed as $a^b$ where $b > 1$.

28. Collatz Sequence: Calculates the stopping time (steps to reach 1) for the $3n+1$ problem.

29. Polygonal Numbers: Generates the $n$-th $s$-gonal number using the formula $\frac{n^2(s-2) - n(s-4)}{2}$.

30. Carmichael Number Check: Identifies pseudoprimes that satisfy Fermat's Little Theorem for all bases coprime to $n$.

31. Miller-Rabin Primality Test: Implements the probabilistic primality test with $k$ rounds for high accuracy.

32. Pollard's Rho Algorithm: Implements integer factorization using Floyd's cycle-finding algorithm.

33. Riemann Zeta Approximation: Numerical approximation of the Riemann Zeta function $\zeta(s)$.

34. Partition Function: Calculates the number of ways to partition $n$ using Euler's Pentagonal Number Theorem.

# Technology Stack

1. Language: Python 3.13.7

2. Performance Modules: time, sys, tracemalloc

3. Math Modules: math, collections

4. Concepts Applied: Dynamic Programming, Number Theory, Cryptography, Complexity Analysis ($O(\log n)$, $O(\sqrt{n})$).

# Skills required

1. Algorithmic Optimization: Transitioning naive $O(n)$ solutions to optimized $O(\sqrt{n})$ or $O(\log n)$ algorithms.

2. Benchmarking: Practical application of profiling tools to measure Wall Time and Peak Memory usage.

3. Mathematical Implementation: Translating complex mathematical theorems (CRT, Euler's Criterion, PNT) into functional code.

4. Edge Case Handling: Managing overflows, negative inputs, and non-coprime conditions robustly.

# Challenges and Solutions

1. Algorithmic Complexity: Transitioning from naive $O(n)$ solutions to $O(\sqrt{n})$ for divisor problems and $O(\log n)$ for modular arithmetic was crucial for handling large inputs without timeouts.

2. Memory Management: Implementing generators and in-place calculations to keep memory usage minimal, as verified by tracemalloc.

3. Mathematical Constraints: Handling edge cases like non-coprime inputs for Modular Inverse or ensuring accurate integer division (//) for geometric formulas.

4. Large Integer Handling: Leveraging Python's automatic large integer support while ensuring algorithm efficiency for cryptographic primitives like RSA (Modular Exponentiation).
