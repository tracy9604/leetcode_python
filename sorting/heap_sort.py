class HeapSort:
    def implement_heap_sort(self, lst: list[int]) -> None:

        def max_heapify(heap_size: int, index: int) -> None:
            left, right = 2 * index + 1, 2 *index + 2
            largest = index

            if left < heap_size and lst[left] > lst[largest]:
                largest = left
            if right < heap_size and lst[right] > lst[largest]:
                largest = right

            if largest != index:
                lst[index], lst[largest] = lst[largest], lst[index]
                max_heapify(heap_size, largest)

        # heapify original lst
        for i in range(len(lst)//2 -1, -1, -1):
            max_heapify(len(lst), i)

        for i in range(len(lst)-1, 0, -1):
            lst[i], lst[0] = lst[0], lst[i]
            max_heapify(i, 0)

    def test_implement_heap_sort(self):
        lst = [4,1,3,2]
        print(self.implement_heap_sort(lst))