from inspect import trace


class ArrayProblems:
    # https://leetcode.com/problems/container-with-most-water/description
    def find_max_area(self, height: list[int]) -> int:
        left, right = 0, len(height) - 1
        max_area = 0

        while left < right:
            curr_area = (right - left) * min(height[left], height[right])
            max_area = max(max_area, curr_area)

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area

    # https://leetcode.com/problems/remove-duplicates-from-sorted-array/description
    def remove_duplicate_in_sorted_array(self, nums: list[int]) -> int:
        j = 0
        for i in range(len(nums)):
            if nums[i] != nums[j]:
                j += 1
                nums[j] = nums[i]

        return j + 1

    # https://leetcode.com/problems/remove-element/description
    def remove_element(self, nums: list[int], val: int) -> int:
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        return k

    # https://leetcode.com/problems/next-permutation
    def next_permutation(self, nums: list[int]) -> None:
        n = len(nums)

        i = next((i for i in range(n - 2, -1, -1) if nums[i] < nums[i + 1]), -1)

        if ~i:
            j = next(j for j in range(n - 1, i, -1) if nums[j] > nums[i])
            nums[i], nums[j] = nums[j], nums[i]

        nums[i + 1:] = nums[i + 1:][::-1]

    # https://leetcode.com/problems/search-in-rotated-sorted-array
    def search_in_rotated_sorted_array(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left < right:
            mid = (left + right) // 2

            # if mid is in the left part
            if nums[0] <= nums[mid]:
                if nums[0] <= target <= nums[mid]:
                    right = mid
                else:
                    left = mid + 1
            else:
                # if mid is in the right part
                if nums[mid] < target <= nums[-1]:
                    left = mid + 1
                else:
                    right = mid
        return left if nums[left] == target else -1

    # https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array
    def search_range(self, nums: list[int], target: int) -> list[int]:

        def binary_search(nums: list[int], target: int, is_search_left: bool) -> int:
            left, right = 0, len(nums) - 1

            idx = -1

            while left <= right:
                mid = (left + right) // 2

                if nums[mid] < target:
                    left = mid + 1
                elif nums[mid] > target:
                    right = mid - 1
                else:
                    idx = mid

                    if is_search_left:
                        right = mid - 1
                    else:
                        left = mid + 1
            return idx

        leftmost = binary_search(nums, target, True)
        rightmost = binary_search(nums, target, False)

        return [leftmost, rightmost]

    # https://leetcode.com/problems/search-insert-position
    def search_insert_pos(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums)

        while left < right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid

        return left

    # https://leetcode.com/problems/valid-sudoku/description
    def is_valid_sudoku(self, board: list[list[str]]) -> bool:
        #     create tracking lists for rows, cols and boxes
        rows = [[False] * 9 for _ in range(9)]
        cols = [[False] * 9 for _ in range(9)]
        boxes = [[False] * 9 for _ in range(9)]

        #     Iterate over each cell in matrix
        for i in range(9):
            for j in range(9):

                # Skip if the cell is empty
                if board[i][j] == '.':
                    continue
                # convert the cell's value into integer and index
                num = int(board[i][j]) - 1

                # Calculate box index for cell (i,j)
                box_index = i // 3 * 3 + j // 3

                if rows[i][num] or cols[j][num] or boxes[box_index][num]:
                    return False

                rows[i][num] = True
                cols[j][num] = True
                boxes[box_index][num] = True
        return True

    # https://leetcode.com/problems/check-if-every-row-and-column-contains-all-numbers/
    def check_valid(self, matrix: list[list[int]]) -> bool:
        n = len(matrix)
        rows = [[False] * n for _ in range(n)]
        cols = [[False] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                num = int(matrix[i][j]) - 1

                if rows[i][num] or cols[j][num]:
                    return False

                rows[i][num] = True
                cols[j][num] = True

        return True

    # https://leetcode.com/problems/matrix-diagonal-sum/
    def diagonal_sum(self, mat: list[list[int]]) -> int:
        total_sum = 0
        n = len(mat)

        for i, row in enumerate(mat):
            j = n - i - 1

            total_sum += row[i]
            if i != j:
                total_sum += row[j]
        return total_sum

    # https://leetcode.com/problems/combination-sum
    def combination_sum(self, candidates: list[int], target: int) -> list[list[int]]:
        def dfs(index: int, current_sum: int) -> None:
            if current_sum == 0:
                combinations.append(combinations_so_far[:])
                return

            if index >= len(candidates) or current_sum < candidates[index]:
                return

            dfs(index + 1, current_sum)
            combinations_so_far.append(candidates[index])
            dfs(index, current_sum - candidates[index])
            combinations_so_far.pop()

        candidates.sort()
        combinations_so_far = []
        combinations = []

        dfs(0, target)
        return combinations

    # https://leetcode.com/problems/combination-sum-ii/
    def combination_sum_2(self, candidates: list[int], target: int) -> list[list[int]]:

        def dfs(index: int, current_sum: int) -> None:
            if current_sum == 0:
                combinations.append(combinations_so_far[:])
                return

            if index >= len(candidates) or current_sum < candidates[index]:
                return

            for i in range(index, len(candidates)):
                if i > index and candidates[i] == candidates[i - 1]:
                    continue

                combinations_so_far.append(candidates[i])
                dfs(i + 1, current_sum - candidates[i])
                combinations_so_far.pop()

        candidates.sort()
        combinations_so_far = []
        combinations = []
        dfs(0, target)
        return combinations

    # https://leetcode.com/problems/jump-game-ii
    def jump_game_2(self, nums: list[int]) -> int:
        jump_count = max_reach = last = 0

        for index, value in enumerate(nums[:-1]):
            max_reach = max(max_reach, index + value)

            if last == index:
                jump_count += 1
                last = max_reach
        return jump_count

    # https://leetcode.com/problems/jump-game/
    def jump_game(self, nums: list[int]) -> bool:
        max_reach = 0

        for index, value in enumerate(nums):
            if max_reach < index:
                return False

            max_reach = max(max_reach, index + value)

        return True

    # https://leetcode.com/problems/permutations
    def permute(self, nums: list[int]) -> list[list[int]]:

        def backtrack(index: int) -> None:
            if index == len_nums:
                permutations.append(current_permutation[:])
                return

            for j in range(len_nums):
                if not visited[j]:
                    visited[j] = True
                    current_permutation[index] = nums[j]
                    backtrack(index + 1)
                    visited[j] = False

        len_nums = len(nums)
        visited = [False] * len_nums
        current_permutation = [0] * len_nums
        permutations = []
        backtrack(0)
        return permutations

    # https://leetcode.com/problems/permutations-ii/
    def permute_unique(self, nums: list[int]) -> list[list[int]]:

        def backtrack(index: int) -> None:
            if index == len_nums:
                permutations.append(current_permutation[:])
                return

            for j in range(len_nums):
                if visited[j] or (j > 0 and not visited[j - 1] and nums[j] == nums[j - 1]):
                    continue

                visited[j] = True
                current_permutation[index] = nums[j]
                backtrack(index + 1)
                visited[j] = False

        nums.sort()
        len_nums = len(nums)
        current_permutation = [0] * len_nums
        visited = [False] * len_nums
        permutations = []
        backtrack(0)
        return permutations

    # https://leetcode.com/problems/rotate-image
    def rotate(self, matrix: list[list[int]]) -> None:
        size = len(matrix)

        #     perform a vertical flip of the matrix
        for i in range(size // 2):
            for j in range(size):
                matrix[i][j], matrix[size - i - 1][j] = matrix[size - i - 1][j], matrix[i][j]
        #     perform a diagonal flip of the matrix
        for i in range(size):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # https://leetcode.com/problems/determine-whether-matrix-can-be-obtained-by-rotation/
    def find_rotation(self, mat: list[list[int]], target: list[list[int]]) -> bool:

        def rotate(matrix: list[list[int]]) -> None:
            size_mat = len(matrix)

            for i in range(size_mat//2):
                for j in range(size_mat):
                    matrix[i][j], matrix[size_mat-i-1][j] = matrix[size_mat-i-1][j], matrix[i][j]

            for i in range(size_mat):
                for j in range(i):
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for _ in range(4):
            rotate(mat)
            if mat == target:
                return True

        return False


    def call_method(self):
        mat = [[0,1],[1,1]]
        target = [[1,0],[0,1]]
        print(self.find_rotation(mat, target))
