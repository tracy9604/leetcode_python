from collections import Counter, defaultdict
from inspect import trace


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class SortingProblems:
    # https://leetcode.com/problems/majority-element/
    def find_majority_element(self, nums: list[int]) -> int:
        for i in set(nums):
            if nums.count(i) > (len(nums) // 2):
                return i

    # https://leetcode.com/problems/3sum/description/
    def find_3_sum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        triplets = []
        for i in range(0, len(nums) - 2):
            if nums[i] > 0:
                break

            if i > 0 and nums[i] == nums[i - 1]:
                continue

            left, right = i + 1, len(nums) - 1
            while left < right:
                curr_sum = nums[i] + nums[left] + nums[right]
                if curr_sum == 0:
                    triplets.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1

                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif curr_sum < 0:
                    left += 1
                else:
                    right -= 1

        return triplets

    # https://leetcode.com/problems/two-sum/description/
    def find_2_sum(self, nums: list[int], target: int) -> list[int]:
        rs = []
        dictNums = {}
        for i, x in enumerate(nums):
            y = target - x
            if y in dictNums:
                rs.append(i)
                rs.append(dictNums[y])
            dictNums[x] = i
        return rs

    # https://leetcode.com/problems/4sum/
    def find_4_sum(self, nums: list[int], target: int) -> list[list[int]]:
        if len(nums) < 4:
            return []

        nums.sort()

        quadruplets = []
        for i in range(0, len(nums) - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            tmp_target = target - nums[i]
            for j in range(i + 1, len(nums) - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                left, right = j + 1, len(nums) - 1
                while left < right:
                    curr_sum = nums[j] + nums[left] + nums[right]
                    if curr_sum == tmp_target:
                        quadruplets.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right and nums[left] == nums[left - 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                    elif curr_sum < tmp_target:
                        left += 1
                    else:
                        right -= 1

        return quadruplets

    # https://leetcode.com/problems/4sum-ii/
    def find_4_sum_ii(self, nums1: list[int], nums2: list[int], nums3: list[int], nums4: list[int]) -> int:
        pre_pairwise = Counter(a + b for a in nums1 for b in nums2)
        count = 0
        for c in nums3:
            for d in nums4:
                count += pre_pairwise[-(c + d)]
        return count

    # https://leetcode.com/problems/3sum-closest/description/
    def find_3_sum_closet(self, nums: list[int], target: int) -> int:
        nums.sort()

        rs = float('inf')
        for i in range(0, len(nums) - 2):
            left, right = i + 1, len(nums) - 1
            while left < right:
                curr_sum = nums[i] + nums[left] + nums[right]
                if curr_sum == target:
                    return target

                if abs(curr_sum - target) < abs(rs - target):
                    rs = curr_sum

                if curr_sum < target:
                    left += 1
                else:
                    right -= 1
        return rs

    # https://leetcode.com/problems/group-anagrams
    def group_anagrams(self, strs: list[str]) -> list[list[str]]:
        anagrams = defaultdict(list)
        for s in strs:
            key = "".join(sorted(s))
            anagrams[key].append(s)
        return list(anagrams.values())

    # https://leetcode.com/problems/valid-anagram/
    def is_anagram(self, s: str, t: str) -> bool:
        return "".join(sorted(s)) == "".join(sorted(t))

    # https://leetcode.com/problems/find-all-anagrams-in-a-string/description/
    def find_anagrams(self, s: str, p: str) -> list[int]:
        x = len(s)
        y = len(p)
        if x < y:
            return []

        rs = []
        p_counter = Counter(p)
        s_window_counter = Counter(s[:y-1])

        for i in range(y-1, x):
            s_window_counter[s[i]] += 1

            if p_counter == s_window_counter:
                rs.append(i-y+1)

            s_window_counter[s[i-y+1]] -= 1
            if s_window_counter[s[i-y+1]] == 0:
                del s_window_counter[s[i-y+1]]

        return rs

    # https://leetcode.com/problems/permutation-in-string/
    def check_permutation_in_string(self, s1: str, s2: str) -> bool:
        x = len(s1)
        y = len(s2)

        if x > y:
            return False

        s1_counter = Counter(s1)
        s2_window_counter = Counter(s2[:x-1])

        for i in range(x-1, y):
            s2_window_counter[s2[i]] += 1

            if s1_counter == s2_window_counter:
                return True

            s2_window_counter[s2[i-x+1]] -= 1
            if s2_window_counter[s2[i-x+1]] == 0:
                del s2_window_counter[s2[i-x+1]]
        return False

    # https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
    def find_length_of_longest_substring(self, s: str) -> int:
        if len(s) == 0:
            return 0

        longest_str = 0
        s_window = Counter(s[0])
        start = 0
        for i in range(1, len(s)):
            s_window[s[i]] += 1

            if s_window[s[i]] > 1:
                longest_str = max(longest_str, i - start + 1) 
            s_window[s[i-start+1]] -= 1
            if s_window[s[i-start+1]] == 0:
                del s_window[s[i-start+1]]

        return longest_str

    def call_method(self):
        s1 = "ab"
        s2 = "eidbaooo"
        print(self.check_permutation_in_string(s1, s2))
