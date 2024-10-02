class SelectionSort:
    def __init__(self, array):
        self.array = array

    def implement_selection_sort(self, arr: list[int]) -> None:
        for i in range(0, len(arr)):
            min_num = i
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[min_num]:
                    min_num = j
            arr[i], arr[min_num] = arr[min_num], arr[i]

    def sort_color_by_selection_sort(self, nums: list[int]) -> None:
        for i in range(0, len(nums)):
            min_num = i
            for j in range(i+1, len(nums)):
                if nums[j] < nums[min_num]:
                    min_num = j
            nums[i], nums[min_num]= nums[min_num], nums[i]
