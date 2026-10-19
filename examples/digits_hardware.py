# The digit-recognising network in digits_network.py, as it is, in hardware.
#
# It is trained in Python (DigitNetwork.fit), then py2ver builds its forward
# pass, net.predict, unchanged: the network is too big to write out (17,226
# weights, ~17,000 multiply-accumulates per image), so its numpy code becomes
# loops, and the loops a small state machine (sequencer.py): one
# multiply-accumulate a clock, the weights in block-RAM ROMs, the activations
# in block RAM, softmax's division a bit a clock. Formats are chosen from
# example images (16 bits).
#
# With --lanes N, the multiply-accumulate loops run N iterations a clock on N
# multipliers (multipliers=N), each lane with its own bank of each weight
# table (the rows it reads), so the weights take the same block RAM on any
# number of lanes: 8 lanes fit the DE0-Nano at 3,481 clocks per image.
#
#   python examples/digits_hardware.py            # fixed point on the test set, hardware simulated
#   python examples/digits_hardware.py --lanes 8  # eight multiply-accumulates a clock
#   python examples/digits_hardware.py --syn      # + Quartus area and timing (DE0-Nano's EP4CE22)
import contextlib
import io
import os
import sys
import time

import numpy as np

import digits_network as D
from py2ver import Py2ver


if __name__ == "__main__":
    net = D.DigitNetwork()
    with contextlib.redirect_stdout(io.StringIO()):
        net.fit(D.X_train, D.y_train, epochs=30)
    print(f"numpy:       test accuracy {net.accuracy(D.X_test, D.y_test):.1%}")

    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        lanes = int(sys.argv[sys.argv.index("--lanes") + 1]) if "--lanes" in sys.argv else 1
        p = Py2ver(net.predict, samples=list(D.X_train[:200]), width=16, multipliers=lanes)
    print(f"hardware:    built in {time.time() - t:.0f} s, {p.ram_phases} clocks per image "
          f"({p.ram_phases / 50e6 * 1e3:.2f} ms at 50 MHz)")

    # The bit-exact model of the hardware, on every test image
    model = [r[0] for r in p.PY_batch()(list(D.X_test))]
    print(f"fixed point: test accuracy {np.mean(np.array(model) == D.y_test):.1%}, "
          f"same digit as numpy for {np.mean(np.array(model) == net.predict(D.X_test)):.1%}")

    # The hardware itself, simulated (Verilator when it is installed), on a few
    n = 20
    t = time.time()
    hw = [r[0] for r in p.TB_batch()(list(D.X_test[:n]))]
    print(f"simulated:   {n} images in {time.time() - t:.0f} s, same as the model: {hw == model[:n]}")

    if "--syn" in sys.argv[1:]:
        print(p.SYN({"QUARTUS_DIR": os.environ.get("QUARTUS_DIR",
                                                   "/opt/intelFPGA/22.1std/quartus/bin")}))
