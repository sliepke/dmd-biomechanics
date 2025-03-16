import json
import numpy as np
import util
import body


b = body.Body("body1.json")
flexion_angles = np.array([0.0] * 10)
flexion_angles[9] = 45.0
print(b.positions(flexion_angles, 7, 0, 0, 8, 0))
