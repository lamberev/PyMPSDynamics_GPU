"""
A logarithmic iterator.
"""
from dataclasses import dataclass

@dataclass
class Log:
    """
    An iterator for a logarithmically spaced sequence.

    Generates `numsteps + 1` points from `start` to `stop` (inclusive).

    Attributes:
        start (float): The starting value of the sequence. Must be positive.
        stop (float): The ending value of the sequence. Must be positive.
        numsteps (int): The number of steps between start and stop. Must be >= 1.
    """
    start: float
    stop: float
    numsteps: int

    def __post_init__(self):
        if self.start <= 0 or self.stop <= 0:
            raise ValueError("start and stop values must be positive for logarithmic spacing.")
        if self.numsteps < 1:
            raise ValueError("numsteps must be at least 1.")
        self._factor = (self.stop / self.start) ** (1 / self.numsteps)

    def __len__(self) -> int:
        """Returns the total number of points in the sequence."""
        return self.numsteps + 1

    def __getitem__(self, i: int) -> float:
        """Returns the i-th element of the sequence (0-indexed)."""
        if not isinstance(i, int):
            raise TypeError(f"Index must be an integer, not {type(i).__name__}.")
        if not 0 <= i < len(self):
            raise IndexError("Index out of range.")
        
        if i == 0:
            return float(self.start)
        if i == self.numsteps:
            return float(self.stop)
        
        return self.start * (self._factor ** i)

    def __iter__(self):
        """Returns an iterator over the sequence."""
        for i in range(len(self)):
            yield self[i] 