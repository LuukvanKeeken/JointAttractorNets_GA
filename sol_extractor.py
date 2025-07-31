import re

solution_numbers = []

with open("2_2.txt", "r") as file:
    for line in file:
        match = re.search(r"solution (\d+)", line)
        if match:
            solution_numbers.append(int(match.group(1)))

print(solution_numbers)
print(len(set(solution_numbers)))