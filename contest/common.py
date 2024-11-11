from collections import Counter


class ContestProblems:
    def find_x_sum(self, nums: list[int], k: int, x: int) -> list[int]:
        ans = []
        for i in range(len(nums)-k+1):
            sub = nums[i:(i+k)]
            sub_counter = Counter(sub)
            sorted_sub = sorted(sub, key = sub.count, reverse=True)
            sub_ans = 0
            count = x
            for i, elem in  enumerate(sorted_sub):
                if count == 0:
                    break
                sub_ans += elem
                if i > 0 and sorted_sub[i] != sorted_sub[i-1]:
                    count -= 1
            ans.append(sub_ans)

        return ans

    def call_method(self):
        nums = [1,1,2,2,3,4,2,3]
        k = 6
        x = 2
        print(self.find_x_sum(nums, k, x))

