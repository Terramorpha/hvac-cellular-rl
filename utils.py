from dataclasses import dataclass


@dataclass
class ProgressIterator:
    name: str
    elements: list

    def __iter__(self):
        n = len(self.elements)
        for i, x in enumerate(self.elements):
            print(f"\r{self.name} {(i+1)/n:1.2f}", end="")
            yield x


@dataclass
class Progress:
    name: str
    elements: list

    def __enter__(self):
        return ProgressIterator(self.name, [x for x in self.elements])

    def __exit__(self, exc_type, exc_value, traceback):
        print("")
