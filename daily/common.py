import heapq
import math
from collections import deque
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class DailyProblem:

    # https://leetcode.com/problems/make-sum-divisible-by-p
    def min_subarray(self, nums: list[int], p: int) -> int:
        sum_num = sum(nums)
        target = sum_num % p
        if target == 0:
            return 0
        n = len(nums)
        prefix_sum = 0
        min_length = len(nums)
        prefix_map = {0: -1}

        for i, num in enumerate(nums):
            prefix_sum = (prefix_sum + num) % p
            desire_sum = (prefix_sum - target) % p

            if desire_sum in prefix_map:
                min_length = min(min_length, i - prefix_map[desire_sum])

            prefix_map[prefix_sum] = i
        return min_length if min_length < n else -1

    # https://leetcode.com/problems/divide-players-into-teams-of-equal-skill
    def divide_players(self, skill: list[int]) -> int:
        skill.sort()
        left, right = 0, len(skill) - 1
        target_sum = skill[left] + skill[right]
        ans = 0
        while left < right:
            if skill[left] + skill[right] != target_sum:
                return -1
            ans += skill[left] * skill[right]
            left += 1
            right -= 1
        return ans

    # https://leetcode.com/problems/permutation-in-string
    def check_inclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        s1_count = [0] * 26
        s2_count = [0] * 26
        for i in range(len(s1)):
            s1_count[ord(s1[i]) - ord('a')] += 1
            s2_count[ord(s2[i]) - ord('a')] += 1

        if s1_count == s2_count:
            return True

        for i in range(len(s1), len(s2)):
            s2_count[ord(s2[i]) - ord('a')] += 1
            s2_count[ord(s2[i - len(s1)]) - ord('a')] -= 1
            if s1_count == s2_count:
                return True

        return False

    # https://leetcode.com/problems/sentence-similarity-iii
    def are_sentences_similar(self, sentence1: str, sentence2: str) -> bool:
        s1 = sentence1.split(" ")
        s2 = sentence2.split(" ")
        n1 = len(s1)
        n2 = len(s2)

        if len(s1) < len(s2):
            s1, s2 = s2, s1
            n1, n2 = n2, n1

        start_idx = end_idx = 0
        while start_idx < n2 and s1[start_idx] == s2[start_idx]:
            start_idx += 1

        while end_idx < n2 and s1[n1 - 1 - end_idx] == s2[n2 - 1 - end_idx]:
            end_idx += 1

        return start_idx + end_idx >= n2

    # https://leetcode.com/problems/minimum-string-length-after-removing-substrings
    def min_length(self, s: str) -> int:
        while "AB" in s or "CD" in s:
            s = s.replace("AB", "")
            s = s.replace("CD", "")

        return len(s)

    # https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced
    def min_swaps(self, s: str) -> int:
        ans = 0
        for sub in s:
            if sub == '[':
                ans += 1
            elif ans:
                ans -= 1
        return (ans + 1) >> 1

    # https://leetcode.com/problems/minimum-add-to-make-parentheses-valid
    def min_add_to_make_valid(self, s: str) -> int:
        st = []
        for sub in s:
            if sub == '(':
                st.append(sub)
            else:
                if st and st[-1] == '(':
                    st.pop()
                else:
                    st.append(sub)
        return len(st)

    # https://leetcode.com/problems/maximum-width-ramp
    def max_width_ramp(self, nums: list[int]) -> int:
        st = []
        for index, value in enumerate(nums):
            if not st or nums[st[-1]] > value:
                st.append(index)

        max_width = 0
        for i in range(len(nums) - 1, -1, -1):
            while st and nums[st[-1]] <= nums[i]:
                max_width = max(max_width, i - st.pop())

            if not st:
                break

        return max_width

    # https://leetcode.com/problems/the-number-of-the-smallest-unoccupied-chair
    def smallest_char(self, times: list[list[int]], targetFriend: int) -> int:
        num_friends = len(times)
        available_chairs = list(range(num_friends))
        heapq.heapify(available_chairs)
        occupied_chars = []

        for friend_index in range(num_friends):
            times[friend_index].append(friend_index)

        # sort by arrival time
        times.sort()

        for arrival, departure, friend_index in times:
            # free up chars if current time is pass the departure time of any friend
            while occupied_chars and occupied_chars[0][0] <= arrival:
                chair_num = heapq.heappop(occupied_chars)[1]
                heapq.heappush(available_chairs, chair_num)

            #         assign the smallest available char to the current friend
            current_chair = heapq.heappop(available_chairs)

            if friend_index == targetFriend:
                return current_chair

            heapq.heappush(occupied_chars, (departure, current_chair))
        return -1

    # https://leetcode.com/problems/divide-intervals-into-minimum-number-of-groups
    def min_groups(self, intervals: list[list[int]]) -> int:
        min_heap = []

        #     sort based on start_time
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        #     loop through intervals
        for start, end in sorted_intervals:
            #         if the heap is not empty and the smallest end is smaller than the current interval's start
            # that means the interval from heap is not overlapped with the current one
            if min_heap and min_heap[0] < start:
                heapq.heappop(min_heap)

            heapq.heappush(min_heap, end)

        return len(min_heap)

    def min_groups_2(self, intervals: list[list[int]]) -> int:
        start_times = sorted(interval[0] for interval in intervals)
        end_times = sorted(interval[1] for interval in intervals)

        start_ptr, end_ptr = 0,0
        open_intervals = 0
        max_groups = 0
        while start_ptr < len(intervals):
            if start_times[start_ptr] < end_times[end_ptr]:
                open_intervals += 1
                max_groups = max(max_groups, open_intervals)
                start_ptr += 1
            else:
                open_intervals -= 1
                end_ptr += 1
        return max_groups

    # 632. Smallest Range Covering Elements from K Lists

    # https://leetcode.com/problems/maximal-score-after-applying-k-operations
    def max_k_elements(self, nums: list[int], k: int) -> int:
        min_heap = [-value for value in nums]
        heapq.heapify(min_heap)

        total_sum = 0

        for _ in range(k):
    #         pop the smallest value from min_heap then reverse it to the original value
            value = -heapq.heappop(min_heap)
            total_sum += value

            new_val = -math.ceil(value/3)
            heapq.heappush(new_val)
        return total_sum

    # https://leetcode.com/problems/separate-black-and-white-balls
    def min_steps(self, s: str) -> int:
        length = len(s)

        ans = 0
        one_count = 0

        # traverse string from right to left
        for i in reversed(range(length)):
            if s[i] == "1":
                one_count += 1

                ans += (length-i-1) - one_count + 1

        return ans

    # https://leetcode.com/problems/longest-happy-string
    def longest_diverse_string(self, a: int, b: int, c: int) -> str:
        max_heap = []

        if a > 0:
            heapq.heappush(max_heap, [-a, 'a'])
        if b > 0:
            heapq.heappush(max_heap, [-b, 'b'])
        if c > 0:
            heapq.heappush(max_heap, [-c, 'c'])

        result = []
        while max_heap:
            current_char = heapq.heappop(max_heap)
            # check if  the last two characters in the result are the same as the current one
            if len(result) >= 2 and result[-1] == current_char[1] and result[-2] == current_char[1]:
                # if there is no other characters, break the loop to avoid three consecutive characters
                if not max_heap:
                    break

                next_char = heapq.heappop(max_heap)

                # add the next_char to the result and decrease its frequency
                result.append(next_char[1])
                if -next_char[0] > 1:
                    next_char[0] += 1
                    heapq.heappush(max_heap, next_char)

                # push the current_char to the heap for future processing
                heapq.heappush(max_heap, current_char)
            else:
                result.append(current_char[1])
                if -current_char[0] > 1:
                    current_char[0] += 1
                    heapq.heappush(max_heap, current_char)
        return "".join(result)

    # https://leetcode.com/problems/maximum-swap/description/
    def maximum_swap(self, num: int) -> int:
        # convert num into list of digits
        digits = list(str(num))
        length = len(digits)

        max_idx = list(range(length))
        for i in range(length-2, -1, -1):
            if digits[i] <= digits[max_idx[i+1]]:
                max_idx[i] = max_idx[i+1]

        for i in range(length):
            max_idx_item = max_idx[i]
            if digits[i] < digits[max_idx_item]:
                digits[i], digits[max_idx_item] = digits[max_idx_item], digits[i]
                break

        return int(''.join(digits))

    # https://leetcode.com/problems/find-kth-bit-in-nth-binary-string
    def find_kth_bit(self, n: int, k: int) -> str:
        def calculate_length(n: int, length_set: set) -> int:
            # base case: the sequence of length 1 has only bit '0'
            if n == 1:
                return 1

            current_length = 2 * calculate_length(n-1, length_set) + 1
            #  add the middle position '1' to the length_set
            length_set.add(current_length//2 + 1)
            return current_length

        def invert_bit(bit: str) -> str:
            return '1' if bit == '0' else '0'

        if k == 1 or n == 1:
            return '0'

        length_set = set()
        length = calculate_length(n, length_set)

        if k in length_set:
            return '1'

        if k < length // 2:
            # k in the first part
            return self.find_kth_bit(n-1, k)
        else:
            # k in the second half
            # invert the (length -k + 1)th bit of the second sequence
            return invert_bit(self.find_kth_bit(n-1,  length - k +1))

    # https://leetcode.com/problems/split-a-string-into-the-max-number-of-unique-substrings/description/
    def max_unique_split(self, s: str) -> int:
        def dfs(index: int, unique_count: int):
            if index >= len(s):
                nonlocal ans
                ans = max(ans, unique_count)
                return

            for j in range(index+1, len(s)+1):
                if s[index: j] not in visited:
                    visited.add(s[index: j])
                    dfs(j, unique_count+1)
                    visited.remove(s[index:j])
        ans = 0
        visited = set()
        dfs(0,0)
        return ans

    # https://leetcode.com/problems/kth-largest-sum-in-a-binary-tree/description/
    def kth_largest_level_sum(self, root: Optional[TreeNode], k: int) -> int:
        level_sums = []
        queue = deque([root])
        while queue:
            current_level_sum = 0
            len_queue = len(queue)
            for _ in range(len_queue):
                node = queue.popleft()
                current_level_sum += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            level_sums.append(current_level_sum)

        if len(level_sums) < k:
            return -1
        level_sums.sort()
        return level_sums[k]

    # https://leetcode.com/problems/flip-equivalent-binary-trees/description/
    def flip_equivalent(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def dfs(node1: TreeNode, node2: TreeNode) -> bool:
            if node1 is None and node2 is None:
                return True
            if node1 is None or node2 is None or node1.val != node2.val:
                return False

            case1 = dfs(node1.left,node2.left) and dfs(node2.right, node2.right)
            case2 = dfs(node1.left,node2.right) and dfs(node1.right, node2.left)
            return case1 or case2

        return dfs(root1, root2)

    # https://leetcode.com/problems/longest-square-streak-in-an-array/description/
    def longest_square_streak(self, nums: list[int]) -> int:
        nums_set = set(nums)
        ans = -1

        for num in nums:
            counter = 0
            while num in nums_set:
                num *= num
                counter += 1
            if counter > 1:
                ans = max(ans, counter)

        return ans

    # https://leetcode.com/problems/maximum-number-of-moves-in-a-grid/description/
    def max_moves(self, grid: list[list[int]]) -> int:
        directions = ((-1,1), (0,1), (1,1))

        num_rows, num_cols = len(grid), len(grid[0])
        queue = deque((row, 0) for row in range(num_rows))

        distance = [[0] * num_cols for _ in range(num_rows)]
        max_moves = 0

        while queue:
            i, j = queue.popleft()
            for delta_row, delta_col in directions:
                new_row, new_col = i + delta_row, j + delta_col

                # check if the new position is in the bounds and if moving there increases the distance
                if 0 <= new_row < num_rows and 0 <= new_col < num_cols and grid[new_row][new_col] > grid[i][j] and distance[new_row][new_col] < distance[i][j] + 1:
                    distance[new_row][new_col] = distance[i][j] + 1
                    max_moves = max(max_moves,distance[new_row][new_col])
                    queue.append((new_row, new_col))

        return max_moves

    # https://leetcode.com/problems/minimum-number-of-removals-to-make-mountain-array/description/
    def minimum_mountain_removals(self, nums: list[int]) -> int:
        length = len(nums)

        # init dp array for the longest increased sequence
        left_lis = [1] * length
        # init dp array for the longest decreased sequence
        right_lis = [1] * length

        # populate the left_lis
        for i in range(1, length):
            for j in range(i):
                if nums[i] > nums[j]:
                    left_lis[i] = max(left_lis[i], left_lis[j]+1)

        # populate the right_lis
        for i in range(length-2, -1,-1):
            for j in range(i+1, length):
                if nums[i] > nums[j]:
                    right_lis[i] = max(right_lis[i], right_lis[j] + 1)

        # calculate the max length of the bitonic sequence
        max_bitonic_length = 0
        for left, right in zip(left_lis, right_lis):
            if left > 1 and right > 1:
                max_bitonic_length = max(max_bitonic_length, left+right-1)

        return length - max_bitonic_length

    # https://leetcode.com/problems/delete-characters-to-make-fancy-string/description/
    def make_fancy_string(self, s: str) -> str:
        rs = []
        for char in s:
            if len(rs) >= 2 and rs[-1] ==char and rs[-2] == char:
                continue
            rs.append(char)

        return "".join(rs)

    # https://leetcode.com/problems/string-compression-iii/description/
    def compress_string(self, word: str) -> str:
        prefix = []
        ans = ""
        for i in range(len(word)):
            if len(prefix) == 0:
                prefix.append(word[i])
                continue
            if prefix[-1] == word[i] and len(prefix) < 9:
                prefix.append(word[i])
            else:
                ans += str(len(prefix)) + prefix[-1]
                prefix = [word[i]]
        if len(prefix) > 0:
            ans += str(len(prefix)) + prefix[-1]
        return ans

    def call_method(self):
        print("start")
        s = "aaaaaaaaaaaaaabb"
        print(self.compress_string(s))
        print("end")
