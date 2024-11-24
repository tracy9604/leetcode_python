import bisect
import math
from bisect import bisect_left
from inspect import trace
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BinarySearchProblem:
    # conditions to apply binary search technique
    # 1. the data structure must be sorted
    # 2. access to any element of the data structure should take constant time
    # 3. Key Difference Between the Two:
    # Criteria	           | left <= right	                                | left < right
    # Purpose	           | Search for an exact match.	                    | Search for a boundary or range.
    # Stopping Condition   | Stops after checking left == right.	        | Stops when left == right.
    # Pointer Updates	   | Both pointers must move to avoid
    #                        infinite loops (e.g., right = mid - 1).	    | No -1 or +1 adjustments needed because the loop stops before invalid states.
    # When It’s Used	   | Finding exact matches.	->
    #                        check every element, including the last one     | Finding boundaries or valid regions.
    def search_matrix(self, matrix: list[list[int]], target: int) -> bool:
        tmp_matrix = []
        for row in range(len(matrix)):
            tmp_matrix += matrix[row]

        left, right = 0, len(tmp_matrix) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if tmp_matrix[mid] == target:
                return True
            elif tmp_matrix[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return False

    # https://neetcode.io/problems/eating-bananas
    def min_eating_speed(self, piles: list[int], h: int) -> int:
        low, high = 1, max(piles)

        while low < high:
            mid = low + (high - low) // 2
            total_hours = sum(math.ceil(pile / mid) for pile in piles)
            if total_hours <= h:
                high = mid
            else:
                low = mid + 1
        return low

    # https://neetcode.io/problems/find-minimum-in-rotated-sorted-array
    def find_min(self, nums: list[int]) -> int:
        # if the array is not rotated
        if nums[0] <= nums[-1]:
            return nums[0]

        left, right = 0, len(nums) - 1

        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] >= nums[0]:
                left = mid + 1
            else:
                right = mid

        return nums[left]

    def find_min_2(self, nums: list[int]) -> int:
        # if the array is already sorted
        if nums[0] <= nums[-1]:
            return nums[0]

        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] > nums[right]:
                left = mid + 1
            elif nums[mid] < nums[right]:
                right = mid - 1
            else:
                right -= 1
        return nums[left]

    # https://neetcode.io/problems/find-target-in-rotated-sorted-array
    def search_target(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2

            # if the mid element is in the sorted part
            if nums[0] <= nums[mid]:
                if nums[0] <= target <= nums[mid]:
                    right = mid
                else:
                    left = mid + 1
            else:
                # if the mid element is in the rotated part
                if nums[mid] < target <= nums[-1]:
                    left = mid + 1
                else:
                    right = mid
        return left if nums[left] == target else -1

    def search_target_with_duplicate_value(self, nums: list[int], target: int) -> bool:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                return True

            if nums[left] == nums[mid] == nums[right]:
                left += 1
                right -= 1

            # if the mid element is in the sorted part
            elif nums[left] <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # if the mid element is in the rotated part
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return False

    # https://neetcode.io/problems/median-of-two-sorted-arrays
    def find_median_sorted_array(self, nums1: list[int], nums2: list[int]) -> float:
        # check the smallest array
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        m, n = len(nums1), len(nums2)
        total = m + n
        half = total // 2
        left, right = 0, m

        while left <= right:
            i = (left + right) // 2  # partition in nums1
            j = half - i  # partition in nums2

            # boundaries
            nums1_left = nums1[i - 1] if i > 0 else float("-inf")
            nums1_right = nums1[i] if i < m else float("inf")
            nums2_left = nums2[j - 1] if j > 0 else float("-inf")
            nums2_right = nums2[j] if j < n else float("inf")

            # correct partition
            if nums1_left <= nums2_right and nums2_left <= nums1_right:
                if total % 2 == 0:
                    return (max(nums1_left, nums2_left) + min(nums1_right, nums2_right)) / 2
                else:
                    return max(nums1_left, nums2_left)
            elif nums1_left > nums2_right:
                right = i - 1
            else:
                left = i + 1

    # https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=problem-list-v2&envId=binary-search
    def search_range(self, nums: list[int], target: int) -> list[int]:
        def find_position(find_left: bool) -> int:
            left, right = 0, len(nums)
            position = -1

            while left <= right:
                mid = left + (right - left) // 2

                if nums[mid] == target:
                    position = mid
                    if find_left:
                        right = mid - 1
                    else:
                        left = mid + 1
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return position

        left_pos = find_position(True)
        right_pos = find_position(False)

        return [left_pos, right_pos]

    # https://leetcode.com/problems/search-a-2d-matrix/description/?envType=problem-list-v2&envId=binary-search
    def search_matrix(self, matrix: list[list[int]], target: int) -> bool:
        num_rows, num_columns = len(matrix), len(matrix[0])

        left, right = 0, num_rows * num_columns - 1

        while left < right:
            mid = left + (right - left) // 2
            row, col = divmod(mid, num_columns)
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] < target:
                left = mid + 1
            else:
                right = mid - 1
        return matrix[left // num_columns][left % num_columns] == target

    # https://leetcode.com/problems/minimum-size-subarray-sum/description/?envType=problem-list-v2&envId=binary-searchd
    def min_subarray_len(self, target: int, nums: list[int]) -> int:
        # compute the prefix sum
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
        min_len = float('inf')
        for i in range(n):
            target_sum = prefix_sum[i] + target
            left = i + 1
            right = n
            while left <= right:
                mid = left + (right - left) // 2

                if prefix_sum[mid] >= target_sum:
                    right = mid - 1
                else:
                    left = mid + 1
            if left <= n:
                min_len = min(min_len, left - i)
        return min_len if min_len != float('inf') else 0

    # https://leetcode.com/problems/count-complete-tree-nodes/description/?envType=problem-list-v2&envId=binary-search
    def count_nodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        # compute the depth
        def get_depth(node: TreeNode, go_left: bool) -> int:
            depth = 0
            while node:
                depth += 1
                node = node.left if go_left else node.right
            return depth

        left_depth = get_depth(root.left, True)
        right_depth = get_depth(root.right, False)

        # if the tree is the completed binary tree
        if left_depth == right_depth:
            return (1 << left_depth) -1
        else:
            return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)

    # https://leetcode.com/problems/search-a-2d-matrix-ii/description/?envType=problem-list-v2&envId=binary-search
    def search_matrix_2(self, matrix: list[list[int]], target: int) -> bool:
        num_rows, num_cols = len(matrix), len(matrix[0])

        # start from the bottom-left position
        row_idx, col_idx = num_rows-1, 0

        while row_idx >= 0 and col_idx < num_cols:
            if matrix[row_idx][col_idx] == target:
                return True

            if matrix[row_idx][col_idx] > target:
                row_idx -= 1
            else:
                col_idx += 1
        return False

    # https://leetcode.com/problems/missing-number/?envType=problem-list-v2&envId=binary-search
    def find_missing_number(self, nums: list[int]) -> int:
        nums.sort()
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right-left)//2
            if nums[mid] == mid:
                left =mid+1
            else:
                right = mid
        return left

    # https://leetcode.com/problems/find-the-duplicate-number
    def find_duplicate(self, nums: list[int]) -> int:
        def is_duplicate_above_x(x: int) -> bool:
            count = sum(num <= x for num in nums)
            return count > x

        left, right= 1, len(nums)-1
        while left < right:
            mid = (left + right + 1)//2
            if is_duplicate_above_x(mid):
                right = mid
            else:
                left = mid +1
        return left

    # https://leetcode.com/problems/longest-increasing-subsequence
    def lenght_of_LIS(self, nums: list[int]) -> int:
        sub= []
        for num in nums:
            idx = bisect.bisect_left(sub, num)
            # if nums[idx] is greater than all element in sub
            if idx == len(sub):
                sub.append(num)
            else:
                sub[idx] = num
        return len(sub)

    # https://leetcode.com/problems/count-of-smaller-numbers-after-self
    def count_smaller(self, nums: list[int]) -> list[int]:
        sorted_list = []
        result = []

        for num in reversed(nums):
            idx = bisect_left(sorted_list, num)

            result.append(idx)

            sorted_list.insert(idx, num)

        return result[::-1]

    # https://leetcode.com/problems/intersection-of-two-arrays-ii
    def intersect(self, nums1: list[int], nums2: list[int]) -> list[int]:
        nums2.sort()
        ans = []
        valid_start = 0

        for num in nums1:
            idx = bisect_left(nums2, num, valid_start)
            if idx < len(nums2) and nums2[idx] == num:
                ans.append(num)
                valid_start = idx + 1
        return ans

    # https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix
    def kth_smallest_element_matrix(self, matrix: list[list[int]], k: int) -> int:
        def check(mid: int) -> bool:
            count = 0
            row, col = size -1, 0 # start from the bottom-left position
            while row >= 0 and col < size:
                if matrix[row][col] <= mid:
                    count += row + 1
                    col += 1
                else:
                    row -= 1
            return count >= k
        size = len(matrix)

        left, right = matrix[0][0], matrix[size-1][size-1]

        while left < right:
            mid = left + (right-left)//2
            if check(mid):
                right = mid
            else:
                left = mid +1
        return left

    def call_method(self):
        nums1 = [1,2,2,1]
        nums2 = [2]
        print(self.intersect(nums1, nums2))
        return
