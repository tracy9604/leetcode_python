from collections import defaultdict
from math import inf
from turtledemo.penrose import inflatedart


class HashTableProblems:
    # https://leetcode.com/problems/longest-substring-without-repeating-characters
    def longest_substring(self, s: str) -> int:
        unique_chars = set()

        left_pointer = 0
        max_length = 0
        for right_pointer, char in enumerate(s):

            while char in unique_chars:
                unique_chars.remove(s[left_pointer])
                left_pointer += 1

            unique_chars.add(char)
            max_length = max(max_length, right_pointer-left_pointer+1)
        return max_length

    # https://leetcode.com/problems/maximum-erasure-value/
    def max_unique_subarray(self, nums: list[int]) -> int:

        unique_arr = set()

        left_pointer = 0
        max_sum = 0

        for right_pointer, elem in enumerate(nums):
            while elem in unique_arr:
                unique_arr.remove(nums[left_pointer])
                left_pointer += 1

            unique_arr.add(elem)
            max_sum = max(max_sum, sum(unique_arr))

        return max_sum

    # https://leetcode.com/problems/minimum-consecutive-cards-to-pick-up/
    def min_card_pick_up(self, cards: list[int]) -> int:
        last_seen = {}
        min_pickup_length = inf

        for index, card_value in enumerate(cards):
            if card_value in last_seen:
                min_pickup_length = min(min_pickup_length, index - last_seen[card_value] + 1)
            last_seen[card_value] = index

        return -1 if min_pickup_length == inf else min_pickup_length

    # https://leetcode.com/problems/integer-to-roman
    def int_to_roman(self, num: int) -> str:
        roman_symbols = ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
        roman_values = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)

        # hold roman string result
        roman_string = []

        for symbol, value in zip(roman_symbols, roman_values):
            # as long as num is greater than or equal to value
            while num >= value:
                num -= value
                roman_string.append(symbol)

        return "".join(roman_string)

    # https://leetcode.com/problems/roman-to-integer/
    def roman_to_int(self, s: str) -> int:
        roman_symbols = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        roman_values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]

        roman_hash = dict(zip(roman_symbols, roman_values))
        num = 0
        for i, sub_str in enumerate(s):
            if i > 0 and roman_hash[sub_str.upper()] > roman_hash[s[i-1].upper()]:
                num += roman_hash[sub_str.upper()] - 2 * roman_hash[s[i-1].upper()]
                continue
            num += roman_hash[sub_str.upper()]
        return num

    # https://leetcode.com/problems/letter-combinations-of-a-phone-number
    def letter_combinations(self, digits: str) -> list[str]:
        if not digits:
            return []

        digit_to_chars = ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']

        combinations = ['']

        for digit in digits:
            letters = digit_to_chars[int(digit) - 2]

            combinations = [prefix + letter for prefix in combinations for letter in letters]

        return combinations

    # https://leetcode.com/problems/set-matrix-zeroes
    def set_zeroes(self, matrix: list[list[int]]) -> None:
        zeroes_hash = {}

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][i] == 0 and not zeroes_hash[[i,j]]:
                    continue
        return

    # https://leetcode.com/problems/binary-watch/
    def read_binary_watch(self, turnedOn: int) -> list[str]:
        return []

    # https://leetcode.com/problems/generate-parentheses/
    def generate_parentheses(self, n: int) -> list[str]:
        def backtrack(open_count, close_count, path):
            if open_count > n or close_count > n or open_count < close_count:
                return
            if open_count == n and close_count == n:
                combinations.append(path)
                return
            backtrack(open_count+1, close_count, path+ '(')
            backtrack(open_count, close_count + 1, path + ')')
        combinations = []
        backtrack(0, 0, '')
        return combinations

    # https://leetcode.com/problems/valid-parentheses/
    def valid_parentheses(self, s: str) -> bool:
        stack = []
        valid_pairs = {'()', '[]', '{}'}

        for char in s:
            if char in '({[':
                stack.append(char)
            elif not stack or stack.pop() + char not in valid_pairs:
                return False
        return not stack

    # https://leetcode.com/problems/combination-sum/
    def combination_sum(self, candidates: list[int], target: int) -> list[list[int]]:
        def backtrack(index: int, current_sum: int):
            if current_sum == 0:
                combinations.append(combinations_so_far[:])
                return
            if index >= len(candidates) or current_sum < candidates[index]:
                return

            backtrack(index+1, current_sum)
            combinations_so_far.append(candidates[index])
            backtrack(index, current_sum - candidates[index])

            combinations_so_far.pop()
        candidates.sort()
        combinations_so_far = []
        combinations = []
        backtrack(0, target)
        return combinations

    # https://leetcode.com/problems/combinations/
    def combine(self, n: int, k: int) -> list[list[int]]:
        def backtrack(num:int):
            if len(current_combination) == k:
                combinations.append(current_combination[:])
                return
            if num > n:
                return
            current_combination.append(num)
            backtrack(num+1)
            current_combination.pop()
            backtrack(num+1)
        combinations = []
        current_combination = []
        backtrack(1)
        return []

    # https://leetcode.com/problems/combination-sum-ii/
    def combination_sum_2(self, candidates: list[int], target: int) -> list[list[int]]:
        def backtrack(index:int, current_sum: int):
            if current_sum == 0:
                combinations.append(combination_so_far[:])
                return
            if index >= len(candidates) or current_sum < candidates[index]:
                return
            for i in range(index, len(candidates)):
                if i > index and candidates[i] == candidates[i-1]:
                    continue
                combination_so_far.append(candidates[index])
                backtrack(i+1, current_sum-candidates[index])
                combination_so_far.pop()

        candidates.sort()
        combinations = []
        combination_so_far = []
        backtrack(0, target)
        return combinations

    # https://leetcode.com/problems/combination-sum-iii/
    def combination_sum_3(self, k: int, n: int) -> list[list[int]]:
        def backtrack(num:int, current_sum: int):
            if len(current_combination) == k and current_sum == 0:
                combinations.append(current_combination[:])
                return
            if num > max_num or current_sum < num:
                return
            current_combination.append(num)
            backtrack(num+1, current_sum-num)
            current_combination.pop()
            backtrack(num+1, current_sum)
        max_num = 9
        combinations = []
        current_combination = []
        backtrack(1, n)
        return combinations

    def call_method(self):
        n = 9
        k = 3
        print(self.combination_sum_3(k,n))