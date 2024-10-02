class BubbleSort:
    def implement_bubble_sort(self, arr: list[int]) -> None:
        has_wrapped = True
        while has_wrapped:
            has_wrapped = False
            for i in range(0, len(arr) - 1):
                if arr[i] > arr[i+1]:
                    arr[i], arr[i+1] = arr[i+1], arr[i]
                    has_wrapped = True

    def height_checker(self, heights: list[int]) -> int:
        has_wrapped = True
        tmp_heights = list(heights)
        while has_wrapped:
            has_wrapped = False
            for i in range(0, len(tmp_heights) - 1):
                if tmp_heights[i] > tmp_heights[i+1]:
                    tmp_heights[i], tmp_heights[i+1] = tmp_heights[i+1], tmp_heights[i]
                    has_wrapped = True

        count = 0
        for i in range(0, len(heights)):
            if heights[i] != tmp_heights[i]:
                count += 1

        return count

    def test_height_checker(self) -> None:
        heights = [1,1,4,2,1,3]
        print(self.height_checker(heights))
