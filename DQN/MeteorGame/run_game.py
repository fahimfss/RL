import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MeteorGame.meteor_img import MeteorImg
from MeteorGame.meteor_arcade import MeteorArcade

import arcade

MODE_AI_ARCADE = 0
MODE_AI_IMG = 1
MODE_HUMAN = 2

MODEL = 'best_16.dat'
SAVE_IMG = True
RUN_MODE = MODE_AI_ARCADE

if RUN_MODE == MODE_AI_IMG:
    mi = MeteorImg(model_path=MODEL, train_mode=False, save_img=SAVE_IMG)
    mi.setup()

    for i in range(9999):
        print(i)
        mi.torch_action()

elif RUN_MODE == MODE_AI_ARCADE:
    window = MeteorArcade(model_path=MODEL, save_img=SAVE_IMG)
    window.setup()
    arcade.run()

elif RUN_MODE == MODE_HUMAN:
    window = MeteorArcade(save_img=SAVE_IMG)
    window.setup()
    arcade.run()

