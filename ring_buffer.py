#!/usr/bin/env python3

import numpy as np
from collections import Sequence


class ContiguousRingBuffer(Sequence, np.lib.mixins.NDArrayOperatorsMixin):
    """
    Numpy class that maintains a contiguous ring buffer;
    due to the additional constraint, maintains twice as large a memory footprint.
    Adaptation from numpy_ringbuffer (https://github.com/eric-wieser/numpy_ringbuffer).
    Also tangentially related to cbuffer (https://github.com/willemt/cbuffer).
    """

    def __init__(self, capacity, dims=(), dtype=np.float32):
        self.buffer_capacity_ = capacity * 2
        self.data_ = np.empty((self.buffer_capacity_,) + dims, dtype)
        self.capacity_ = capacity
        self.left_index_ = 0
        self.right_index_ = 0

    def __array__(self):
        return self.data_[self.left_index_: self.right_index_]

    @property
    def array(self):
        return self.__array__()

    # NOTE(yycho0108): Deliberately disable setter method
    # To force usage of ContiguousRingBuffer.array.
    # @array.setter
    # def array(self, value):
    #    self.data_[self.left_index_: self.right_index_] = value

    @property
    def is_full(self):
        return len(self) == self.capacity_

    @property
    def dtype(self):
        return self.data_.dtype

    @property
    def shape(self):
        return (len(self),) + self.data_.shape[1:]

    @property
    def maxlen(self):
        return self.capacity_

    def extend(self, values):
        # Always only consider last N values.
        values = values[-self.capacity_:]
        di = len(values)

        if self.right_index_ + di > self.buffer_capacity_:
            # Reset to left_index == 0
            # self.data_ = np.roll(self.data_, -self.left_index_, axis=0)
            self.data_[:len(self)] = self.__array__()
            self.right_index_ -= self.left_index_
            self.left_index_ = 0
        self.data_[self.right_index_: self.right_index_ + di] = values
        self.right_index_ += di
        self.left_index_ = max(0, self.right_index_ - self.capacity_)

    def append(self, value):
        return self.extend([value])

    def reset(self):
        self.left_index_ = 0
        self.right_index_ = 0

    def __getattr__(self, *args, **kwargs):
        return self.__array__().__getattr__(*args, **kwargs)

    def __len__(self):
        return self.right_index_ - self.left_index_

    def __getitem__(self, item):
        return self.__array__().__getitem__(item)

    def __setitem__(self, item, value):
        return self.__array__().__setitem__(item, value)

    def __iter__(self):
        return self.__array__().__iter__()

    def __repr__(self):
        return self.__array__().__repr__()


def main():
    from collections import deque
    q1 = deque(maxlen=17)
    q2 = ContiguousRingBuffer(capacity=17, dims=())

    elements = np.random.uniform(size=35)
    print('Ground Truth')
    print(elements[-17:])
    for e in elements:
        q1.append(e)
        q2.append(e)
        print(q2[-5:])

    print('Deque')
    print(q2.__array__().flags['C_CONTIGUOUS'])
    print(list(q1))
    print('ContiguousRingBuffer')
    print(list(q2))

    q2.extend(elements)
    print('ContiguousRingBuffer')
    print(list(q2))
    print(np.random.choice(q2, size=16))
    # print(q2[18])

    a = np.asarray(q2)
    print(q2)
    print(a)
    a += 1
    print(q2)
    print(a)
    a1 = q2.array
    print('a1')
    print(a1)
    a1 += 3
    print(q2)
    print('a1')
    print(a1)
    print(q2)
    print(type(q2))

    print(len(q2))


if __name__ == '__main__':
    main()
