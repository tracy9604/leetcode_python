import math
import sys
from collections import deque, Counter


class SlidingWindowProblems:
    # https://www.geeksforgeeks.org/problems/smallest-subarray-with-sum-greater-than-x5651
    def smallest_sub_with_sum(self, x, arr):
        n = len(arr)
        min_length = 0
        current_sum= 0
        start = 0
        for end in range(n):
            # add the current element to current_sum
            current_sum += arr[end]

            # while the current_sum is greater than x, try to shrink the window
            while current_sum > x:
                min_length= min(min_length, end-start+1)
                current_sum -= arr[start]
                start += 1
        return min_length

    # https://www.geeksforgeeks.org/find-subarray-with-given-sum/
    def sub_array_with_given_sum(self, arr: list[int], n: int, sum: int) -> list[int]:
        start, end = 0,0
        cur_sum = 0

        for i in range(len(arr)):
            cur_sum += arr[i]

            if cur_sum >= sum:
                end = i

                while start < end and cur_sum > sum:
                    cur_sum -= arr[start]
                    start -=1

                if cur_sum == sum:
                    return [start+1, end+1]

        return [-1]

    # https://www.geeksforgeeks.org/problems/smallest-distant-window3132/1?itm_source=geeksforgeeks&itm_medium=article&itm_campaign=practice_card
    def find_smallest_distinct_substring(self, s: str):
        unique_chars = set(s)
        start = 0
        min_length = float('inf')
        cur_window_counter = {}

        for end in range(len(s)):
            cur_window_counter[s[end]] = cur_window_counter.get(s[end], 0) + 1

            while len(unique_chars) == len(cur_window_counter.keys()):
                min_length = min(min_length, end-start+1)

                cur_window_counter[s[start]] -= 1
                if cur_window_counter[s[start]] == 0:
                    del cur_window_counter[s[start]]

                start += 1
        return min_length if min_length != float('inf') else 0

    # https://www.geeksforgeeks.org/longest-sub-array-sum-k/
    def longest_subarray_with_sum_k(self, arr: list[int], k: int) -> int:
        max_length = -math.inf
        start = 0
        current_sum = 0

        for end in range(len(arr)):
            current_sum += arr[end]

            if current_sum >= k:
                while start < end and current_sum > k:
                    current_sum -= arr[start]
                    start += 1

                if current_sum == k:
                    max_length = max(max_length, end - start + 1)
        return max_length

    # https://www.geeksforgeeks.org/find-maximum-minimum-sum-subarray-size-k/
    def max_sum_sub_array_size_k(self,arr:list[int],k: int) -> int:
        cur_sum = sum(arr[:k])
        max_sum = cur_sum

        for end in range(k, len(arr)):
            cur_sum += arr[end] - arr[end-k]
            max_sum = max(max_sum, cur_sum)
        return max_sum

    # https://www.geeksforgeeks.org/length-of-the-longest-substring-without-repeating-characters/
    def longest_sub_str_distinct_chars(self, s: str) -> int:
        start = 0
        max_length = 0
        char_set = set()

        for end in range(len(s)):
            while s[end] in char_set:
                char_set.remove(s[start])
                start += 1

            char_set.add(s[end])
            max_length = max(max_length, end-start+1)

        return max_length

    # https://www.geeksforgeeks.org/count-distinct-elements-in-every-window-of-size-k/
    def count_distinct_each_window(self, k: int, arr: list[int]) -> list[int]:
        if k > len(arr):
            return []

        current_window = arr[:k]
        ans = [len(set(current_window))]

        for i in range(k, len(arr)):
            current_window.append(arr[i])
            current_window.remove(arr[i-k])
            ans.append(len(set(current_window)))
        return ans

    # https://www.geeksforgeeks.org/first-negative-integer-every-window-size-k/
    def first_negative_integer(self, arr: list[int], k: int) -> list[int]:
        ans = []
        queue = deque()

        for i in range(len(arr)):
            if arr[i] < 0:
                queue.append(i)

            # remove indices that are out of the current window
            if queue and queue[0] < i-k+1:
                queue.popleft()

            # start from k-1
            if i >= k-1:
                if queue:
                    ans.append(arr[queue[0]])
                else:
                    ans.append(0)
        return ans

    # https://neetcode.io/problems/buy-and-sell-crypto
    def max_profit(self, prices: list[int]) -> int:
        start = 0
        max_profit = 0

        for end in range(len(prices)):
            while start < end and prices[end] <prices[start]:
                start += 1

            max_profit = max(max_profit, prices[end]-prices[start])
        return max_profit

    # https://neetcode.io/problems/longest-repeating-substring-with-replacement
    def character_replacement(self, s: str, k: int) -> int:
        left = 0
        max_length = 0
        max_count = 0
        frequency = {}

        for right in range(len(s)):
            char = s[right]
            frequency[char] = frequency.get(char, 0) + 1

    #         update max count of frequency in the current window
            max_count = max(max_count, frequency[char])

            # means we need more k replacements for the current window
            while (right-left+1) - max_count > k:
                frequency[s[left]] -= 1
                left += 1

            max_length = max(max_length, right-left+1)
        return max_length

    # https://neetcode.io/problems/permutation-string
    def check_inclusion(self, s1: str, s2: str) -> bool:
        len_s1 = len(s1)
        len_s2 = len(s2)

        s1_counter = Counter(s1)

        for i in range(len_s2 - len_s1+1):
            cur_s2_counter = Counter(s2[i: i + len_s1])
            if s1_counter == cur_s2_counter:
                return True
        return False

    # https://neetcode.io/problems/minimum-window-with-characters
    def min_window(self, s: str, t: str) -> str:
        t_counter = Counter(t)
        t_required= len(t_counter)

        left, right = 0, 0
        cur_count = 0
        cur_window = {}

        min_length = math.inf
        min_left, min_right = 0,0

        while right < len(s):
            char = s[right]
            cur_window[char] = cur_window.get(char, 0) + 1

            if char in t_counter and cur_window[char] == t_counter[char]:
                cur_count += 1

            while left <= right and cur_count == t_required:
                cur_char = s[left]

                if right -left + 1 <min_length:
                    min_length = right-left+1
                    min_left = left
                    min_right = right

                if cur_char in t_counter and cur_window[cur_char] < t_counter[cur_char]:
                    cur_count -= 1

                left += 1
            right += 1
        return "" if min_length == math.inf else s[min_left:min_right+1]

    def call_method(self):
        s1 = "adc"
        s2 = "dcda"
        print(self.check_inclusion( s1, s2))
        print("end")