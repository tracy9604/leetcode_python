import math
from audioop import reverse
from collections import defaultdict
from inspect import trace
from itertools import permutations
from operator import truediv
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TwoPointersProblems:
    # https://leetcode.com/problems/longest-palindromic-substring
    def longest_palindrome_string(self, s: str) -> str:
        max_length = 1
        n = len(s)
        start = 0

        def expand_from_center(left: int, right: int) -> int:
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1

        for i in range(n):
            #     odd string
            odd_len = expand_from_center(i, i)
            if odd_len > max_length:
                max_length = odd_len
                start = i - max_length // 2

            even_len = expand_from_center(i, i + 1)
            if even_len > max_length:
                max_length = even_len
                start = i - (max_length - 1) // 2

        return s[start: start + max_length]

    # https://leetcode.com/problems/palindromic-substrings/
    def count_substrings(self, s: str) -> int:
        n = len(s)

        def expand_from_center(left: int, right: int) -> int:
            count = 0
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
                count += 1
            return count

        total_palindromes = 0
        for i in range(n):
            total_palindromes += expand_from_center(i, i)
            total_palindromes += expand_from_center(i, i + 1)

        return total_palindromes

    # https://leetcode.com/problems/container-with-most-water
    def max_area(self, height: list[int]) -> int:
        ans = 0
        i, j = 0, len(height) - 1
        while i < j:
            current_height = (j - i) * min(height[i], height[j])
            ans = max(ans, current_height)

            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return ans

    # https://leetcode.com/problems/3sum
    def three_sum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        ans = []
        n = len(nums)
        for i, num1 in enumerate(nums):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            left = i + 1
            right = n - 1

            while left < right:
                curr_sum = num1 + nums[left] + nums[right]
                if curr_sum == 0:
                    ans.append([num1, nums[left], nums[right]])
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
        return ans

    #  https://leetcode.com/problems/two-sum/
    def two_sum(self, nums: list[int], target: int) -> list[int]:
        dict_nums = {}

        for i, x in enumerate(nums):
            y = target - x
            if y in dict_nums:
                return [i, dict_nums[y]]
            dict_nums[x] = i
        return []

    # https://leetcode.com/problems/4sum/
    def four_sum(self, nums: list[int], target: int) -> list[list[int]]:
        nums.sort()
        n = len(nums)
        ans = []
        for i, num1 in enumerate(nums):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            for j in range(i + 1, n):
                num2 = nums[j]
                if j > 0 and nums[j] == nums[j - 1]:
                    continue
                left, right = j, n - 1
                while left < right:
                    cur_sum = num1 + num2 + nums[left] + nums[right]
                    if cur_sum == target:
                        ans.append([num1, num2, nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right and nums[left] == nums[left - 1]:
                            left += 1
                        while left < right and nums[right] == nums[right + 1]:
                            right -= 1
                    elif cur_sum < target:
                        left += 1
                    else:
                        right -= 1
        return ans

    # https://leetcode.com/problems/3sum-closest
    def three_sum_closest(self, nums: list[int], target: int) -> int:
        nums.sort()

        closest_ans = 0
        closest = math.inf

        for i in range(len(nums)):
            left, right = i + 1, len(nums) - 1
            while left < right:
                curr_sum = nums[i] + nums[left] + nums[right]
                if curr_sum == target:
                    return 0
                else:
                    if abs(curr_sum - target) < closest:
                        closest_ans = curr_sum
                        closest = abs(curr_sum - target)
                    if curr_sum < target:
                        left += 1
                    else:
                        right -= 1
        return closest_ans

    # https://leetcode.com/problems/remove-nth-node-from-end-of-list
    def remove_nth_from_end(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        slow, fast = dummy, dummy

        for _ in range(n):
            fast = fast.next

        while fast.next:
            slow = slow.next
            fast = fast.next

        if slow and slow.next:
            slow.next = slow.next.next

        return dummy.next

    # https://leetcode.com/problems/swapping-nodes-in-a-linked-list/
    def swap_nodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = head
        fast, slow, begin_node = dummy, dummy, dummy

        for _ in range(k):
            fast = fast.next
            begin_node = begin_node.next

        while fast.next:
            slow, fast = slow.next, fast.next

        tmp = begin_node.val
        begin_node.val = slow.next.val
        slow.next.val = tmp
        return dummy.next

    # https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/
    def delete_middle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        fast, slow = head, dummy
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        if slow and slow.next:
            slow.next = slow.next.next
        return dummy.next

    # https://leetcode.com/problems/remove-duplicates-from-sorted-array
    def remove_duplicates(self, nums: list[int]) -> int:
        cur_idx = 0

        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                continue
            cur_idx += 1
            nums[cur_idx] = nums[i]
        return cur_idx + 1

    # https://leetcode.com/problems/remove-element
    def remove_element(self, nums: list[int], val: int) -> int:
        cur_idx = 0

        for i in range(len(nums)):
            if nums[i] == val:
                continue
            nums[cur_idx] = nums[i]
            cur_idx += 1

        return cur_idx

    # https://leetcode.com/problems/move-zeroes/
    def move_zeroes(self, nums: list[int]) -> None:
        cur_idx = 0
        for i, num in enumerate(nums):
            if num != 0:
                nums[i], nums[cur_idx] = nums[cur_idx], nums[i]
                cur_idx += 1

    # https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string
    def first_occurrence(self, haystack: str, needle: str) -> int:
        n1 = len(haystack)
        n2 = len(needle)
        for i in range(n1):
            s = haystack[i:i + n2] if i + n2 < n1 else haystack[i:]
            if s == needle:
                return i
        return -1

    # https://leetcode.com/problems/next-permutation
    def next_permutation(self, nums: list[int]) -> None:
        pivot = -1
        n = len(nums)

        for i in range(n - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                pivot = i
                break

        if pivot != -1:
            for i in range(n - 1, pivot, -1):
                if nums[i] > nums[pivot]:
                    nums[pivot], nums[i] = nums[i], nums[pivot]

        nums[pivot + 1:] = reversed(nums[pivot + 1:])

    #  https://leetcode.com/problems/permutations/
    def permute(self, nums: list[int]) -> list[list[int]]:
        n = len(nums)

        def backtrack(index: int):
            if index >= n:
                permutations.append(permutation_so_far[:])
                return
            for i in range(n):
                if not visited[i]:
                    visited[i] = True
                    permutation_so_far[index] = nums[i]
                    backtrack(index + 1)
                    visited[i] = False

        permutations = []
        permutation_so_far = [0] * n
        visited = [False] * n
        backtrack(0)
        return permutations

    # https://leetcode.com/problems/permutations-ii/
    def permute_unique(self, nums: list[int]) -> list[list[int]]:
        n = len(nums)
        nums.sort()

        def backtrack(index: int):
            if index >= n:
                permutations.append(permutation_so_far[:])
                return

            for i in range(n):
                if visited[i] or (i > 0 and not visited[i - 1] and nums[i] == nums[i - 1]):
                    continue

                visited[i] = True
                permutation_so_far[index] = nums[i]
                backtrack(index + 1)
                visited[i] = False

        permutations = []
        permutation_so_far = [0] * n
        visited = [False] * n

        return permutations

    # https://leetcode.com/problems/rotate-list
    def rotate_right(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head

        length = 0
        cur_node = head
        while cur_node:
            length += 1
            cur_node = cur_node.next

        k %= length
        if k == 0:
            return head

        fast, slow = head, head
        for _ in range(k):
            fast = fast.next

        while fast.next:
            slow, fast = slow.next, fast.next

        new_head = slow.next
        slow.next = None
        fast.next = head
        return new_head

    # https://leetcode.com/problems/rotate-array/
    def rotate_array(self, nums: list[int], k: int) -> None:
        k %= len(nums)
        nums[:] = nums[-k:] + nums[:-k]

    # https://leetcode.com/problems/sort-colors
    def sort_color(self, nums: list[int]) -> None:
        max_num = max(nums)
        count = [0] * (max_num + 1)

        for num in nums:
            count[num] += 1

        index = 0
        for i in range(len(count)):
            while count[i] > 0:
                nums[index] = i
                count[i] -= 1
                index += 1

    # https://leetcode.com/problems/sort-list/
    def sort_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head

        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next

        # split the list into two halves
        temp = slow.next
        slow.next = None
        left_half, right_half = self.sort_list(head), self.sort_list(temp)
        dummy_node = ListNode()
        cur_node = dummy_node
        while left_half or right_half:
            if left_half.val <= right_half.val:
                cur_node.next = left_half
                left_half = left_half.next
            else:
                cur_node.next = right_half
                right_half = right_half.next
            cur_node = cur_node.next

        cur_node.next = left_half or right_half
        return dummy_node.next

    # https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii
    def remove_duplicates_2(self, nums: list[int]) -> int:
        cur_idx = 0
        for i in range(len(nums)):
            if cur_idx < 2 or nums[i] != nums[i - 2]:
                nums[cur_idx] = nums[i]
                cur_idx += 1

        return cur_idx + 1

    # https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii
    def delete_duplicate(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return head
        dummy_node = ListNode(next=head)
        prev, cur = dummy_node, head

        while cur:
            while cur.next and cur.val == cur.next.val:
                cur = cur.next
            if prev.next == cur:
                prev = cur
            else:
                prev.next = cur.next
            cur = cur.next
        return dummy_node.next

    # https://leetcode.com/problems/partition-list
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        if head is None:
            return head

        smaller_head, greater_head = ListNode(), ListNode()
        smaller_tail = smaller_head
        greater_tail = greater_head
        cur = head
        while cur:
            if cur.val < x:
                smaller_tail.next = cur
                smaller_tail = smaller_tail.next
            else:
                greater_tail.next = cur
                greater_tail = greater_tail.next
            cur = cur.next

        smaller_tail.next = greater_head.next
        greater_tail.next = None

        return smaller_head.next

    # https://leetcode.com/problems/partition-array-according-to-given-pivot/
    def pivot_array(self, nums: list[int], pivot: int) -> list[int]:
        smaller_array = []
        greater_array = []
        equal_array = []
        for num in nums:
            if num < pivot:
                smaller_array.append(num)
            elif num > pivot:
                greater_array.append(num)
            else:
                equal_array.append(pivot)
        return smaller_array + equal_array + greater_array

    # https://leetcode.com/problems/merge-sorted-array
    def merge_sort_array(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        idx_nums1, idx_nums2 = m - 1, n - 1
        idx_ans = m + n - 1
        while idx_nums2 >= 0:
            if idx_nums1 >= 0 and nums1[idx_nums1] > nums2[idx_nums2]:
                nums1[idx_ans] = nums1[idx_nums1]
                idx_nums1 -= 1
            else:
                nums1[idx_ans] = nums2[idx_nums2]
                idx_nums2 -= 1
            idx_ans -= 1

    # https://leetcode.com/problems/merge-two-sorted-lists/
    def merge_two_lists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy_node = ListNode()
        cur = dummy_node
        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next

        if list1:
            cur.next = list1
        if list2:
            cur.next = list2

        return dummy_node.next

    # https://leetcode.com/problems/valid-palindrome
    def is_valid_palindrome(self, s: str) -> bool:
        s = s.lower()
        left, right = 0, len(s) - 1
        while left <= right:
            if not s[left].isalpha() and not s[left].isdigit():
                left += 1
                continue
            if not s[right].isalpha() and not s[right].isdigit():
                right -= 1
                continue

            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return False
        return True

    # https://leetcode.com/problems/valid-palindrome-ii/
    def valid_palindrome(self, s: str) -> bool:
        def is_palindrome(left: int, right: int) -> bool:
            while left <= right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True

        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return is_palindrome(left, right - 1) or is_palindrome(left + 1, right)
            left += 1
            right -= 1
        return True

    # https://leetcode.com/problems/linked-list-cycle
    def has_cycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                return True
        return False

    # https://leetcode.com/problems/reorder-list
    def reorder_list(self, head: Optional[ListNode]) -> None:
        #     find the middle node
        fast, slow = head, head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        #  split the list into two halves
        second_half = slow.next
        slow.next = None

        #     reverse the second half
        previous = None
        current = second_half

        while current:
            temp = current.next
            current.next = previous
            previous, current = current, temp
        #     merge two halves
        first_half = head
        second_half = previous
        while second_half:
            temp1 = first_half.next
            temp2 = second_half.next

            first_half.next = second_half
            second_half.next = temp1

            first_half, second_half = temp1, temp2

    # https://leetcode.com/problems/reverse-words-in-a-string
    def reverse_words(self, s: str) -> str:
        s_arr = s.split()
        left, right = 0, len(s_arr) - 1
        while left <= right:
            s_arr[left], s_arr[right] = s_arr[right], s_arr[left]
            left += 1
            right -= 1

        return " ".join(s_arr)

    # https://leetcode.com/problems/intersection-of-two-linked-lists
    def get_intersection_node(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        return None

    # https://leetcode.com/problems/compare-version-numbers
    def compare_version(self, version1: str, version2: str) -> int:
        left, right = 0, 0
        n1, n2 = len(version1), len(version2)

        while left < n1 or right < n2:
            ver1_part = ""
            while left < n1 and version1[left] != ".":
                ver1_part += version1[left]
                left += 1

            ver2_part = ""
            while right < n2 and version2[right] != ".":
                ver2_part += version2[right]
                right += 1

            if len(ver1_part) == 0:
                ver1_part = "0"
            if len(ver2_part) == 0:
                ver2_part = "0"

            if int(ver1_part) < int(ver2_part):
                return -1
            elif int(ver1_part) > int(ver2_part):
                return 1
            else:
                left += 1
                right += 1
        return 0

    # https://leetcode.com/problems/two-sum-iv-input-is-a-bst/
    def find_target(self, root: Optional[TreeNode], k: int) -> bool:
        def dfs(node: TreeNode):
            if node is None:
                return False

            if k - node.val in visited:
                return True

            visited.add(node.val)
            return dfs(node.left) or dfs(node.right)

        visited = set()
        return dfs(root)

    # https://leetcode.com/problems/happy-number
    def is_happy(self, n: int) -> bool:
        def get_next_number(x: int):
            total_sum = 0
            while x:
                x, digit = divmod(x, 10)
                total_sum += digit * digit
            return total_sum

        slow, fast = n, get_next_number(n)
        while slow != fast:
            slow = get_next_number(slow)
            fast = get_next_number(get_next_number(fast))
        return slow == 1

    # https://leetcode.com/problems/ugly-number/
    def is_ugly_number(self, n: int) -> bool:
        if n < 1:
            return False

        prime_factors = [2, 3, 5]
        for factor in prime_factors:
            while n % factor == 0:
                n //= factor

        return n == 1

    # https://leetcode.com/problems/ugly-number-ii/
    def ugly_number_2(self, n: int) -> int:
        ugly_numbers = [1] * n

        index2, index3, index5 = 0, 0, 0

        for i in range(1, n):
            next2 = ugly_numbers[index2] * 2
            next3 = ugly_numbers[index3] * 3
            next5 = ugly_numbers[index5] * 5

            ugly_numbers[i] = min(next2, next3, next5)

            if ugly_numbers[i] == next2:
                index2 += 1
            if ugly_numbers[i] == next3:
                index3 += 1
            if ugly_numbers[i] == next5:
                index5 += 1


        return ugly_numbers[n - 1]

    # https://leetcode.com/problems/palindrome-linked-list
    def is_palindrome_linked_list(self, head: Optional[ListNode]) -> bool:
        # find the middle node
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next

        # separate the list into two halves
        second_head = slow.next
        slow.next = None

        # reverse the second half
        dummy = None
        cur_node = second_head
        while cur_node:
            tmp = cur_node.next
            cur_node.next = dummy
            dummy = cur_node
            cur_node = tmp

        second_head = dummy

        # compare two lists
        while head and second_head:
            if head.val != second_head.val:
                return False
            head = head.next
            second_head = second_head.next

        return True if head is None and second_head is None else False

    # https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/
    def pair_sum(self, head: Optional[ListNode]) -> int:
        if head is None:
            return 0
        # find the middle node
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next

        ans = 0
        second_head = slow.next
        slow.next = None

        # reverse the second list
        prev = None
        cur_node = second_head
        while cur_node:
            tmp = cur_node.next
            cur_node.next = prev
            prev = cur_node
            cur_node = tmp

        # compare two lists
        while prev:
            ans = max(ans, head.val + prev.val)
            head, prev = head.next, prev.next

        return ans

    # https://leetcode.com/problems/find-the-duplicate-number
    def find_duplicate(self, nums: list[int]) -> int:
        return 0

    def call_method(self):
        head = ListNode(1, ListNode(2, ListNode(2, ListNode(1))))
        print(self.is_palindrome_linked_list(head))
        return
