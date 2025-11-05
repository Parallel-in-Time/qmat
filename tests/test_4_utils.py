from time import sleep

from qmat.utils import Timer

def testTimer():
    with Timer("test1"):
        pass

    clock = Timer("test2")
    clock.start()
    sleep(0.1)
    clock.stop()
    assert clock.tWall >= 0.1
