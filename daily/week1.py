import string


class Week1:
    def get_lucky(self, s: str, k: int) -> int:
        sNum = []
        for x in s:
            sNum.append(string.ascii_lowercase.index(x)+1)

        s1 = "".join(map(str, sNum))
        num = 0
        while k > 0:
            num = sum(int(x) for x in s1)
            s1 = str(num)
            k -= 1

        return num

    def test_week_1(self):
        s = "zbax"
        k = 2
        print(self.get_lucky(s, k))