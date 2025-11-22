# 34-Python-Code-
A collection of 34 Python Programs covering basic to intermediate concepts.

This repository contains a collection of python functions and implementing various concepts from number theory, discrete mathematics, and computational algorithms. Each function is designed to be efficient, and the included source code often includes performance metrics (execution time and memory usage) for benchmarking purposes. 


# Project Highlights

1. diverse mathematical function: from baisc arithmetic properties(factorial, mean of digits) to advanced number theory (modular arithmetic, carmichael numbers, riemann zeta approximation).
2. performance measurement: most code blocks include utility code to measure execution time and memory usage (using time and sys/tracemalloc).
3. fundamental algorithms: implementation of core algorithms like modular exponentiation, extended euclidean algorithm, and pollard's rho for factorization.

# Prerequisites
python3.x installed on your system.

# Function

1. factorial(n) - calculates n! (factorial).
2. is_palindrome(n) - checks if a number is a palindrome.
3. mean_of_digits(n) - returns the average of all digits in n.
4. digital_root(n) - calculates the digital root of a number 
5. collatz_length(n) - returns the steps to reach 1 in the collatz sequence.
6. is_abundant(n) - checks if a number is abundant (∑proper divisors>n).
7. is_deficient(n) - checks if a number is deficient (∑proper divisors<n).
8. aliquot_sum(n) - returns the sum of proper divisors.
9. count_divisors(n) - counts the number of proper divisors d(n).
10. are_amicable(a, b) - checks if two numbers are an amicable pair.
11. is_ptime_power(n) - checks if n = p^𝑘 where p is a prime and k>_1.
12. is_mersenne_prime(p) - checks if 2^p - 1 is a mersenne prime(using Lucas-lehmer test logic).
13. twin_primes(limit) - generates twin prime pairs up to a given limit.
14. prime_factors(n) - returns the list of all prime factors of n.
15. count_distinct_prime_factors(n) - returns the count of unique prime factors.
16. is_harshad(n) - checks if n is a harshad (niven) number.
17. is_automorphic(n) - checks if n is automorphic (n^2 ends with n).
18. is_pronic(n) - checks if n pronic (n = k(k+1)).
19. is_highly_composite(n) -checks if n has more divisors than any smaller number.
20. is_carmichael(n) - checks if n is a charmichael number (a pseudoprime).
21. is_fibonacci_prime(n) - checks if a number is both a fibonacci number and prime.
22. lucas_sequence(n) - generates the first n lucas numbers.
23. is_perfect_power(n) - checks if n =a^b where a>0, b>1.
24. polygonal_number(s, n) - calculates the n-th s-gonal number.
25. mod_exp(base, exp, mod) - modular exponentiation:(base^exponent) (mod mod).
26. mod_inverse(a, m) - modular multiplicative inverse:x s.t.(a.x) = 1 (mod m).
27. crt (remainders, moduli) - chinese remainder theorem solver.
28. is_quadratic_residue(a, p) - checks if x^2 = a (mnod p) has a solution (using euler's criterion).
29. order_mod(a, n) - finds the multiplicative order of a modulo n.
30. is_prime_mr(n, k) - miler-rabin primality test(probaiblistic).
31. pollard_rho_manual(n) - pollard's rho algorithm for intger factorization.
32. zeta_approx(s, terms) - approximates the riemann zeta function.
33. partition_function(n) - caculates the partition function p(n).

# Dependencies

1. import time
2. import sys
3. import math
4. import tracemalloc (for detailed memory tracking)
