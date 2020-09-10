import random
import os
import time
import numpy as np
from PIL import Image
from arcade.sprite import Sprite

from DQN_Dueling.model import DuelingDQN
import torch

import MeteorGame.parameters as PRM
from MeteorGame.meteor_arcade import Player, Meteor

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


# -Generate pillow images as game states
# -High level of caching is done so that
# during runtime, calculation is minimum
# -Can also save the state images in disk.
# The images can be used to make a video of the agent
# playing the game.
class MeteorImages:
    def __init__(self, save_img=False):
        self.save_img = save_img
        self.count = 0
        self.image_cache = {}
        self.sprite_cache = {}
        for idx, image_name in enumerate(PRM.METEORS_IMAGES):
            img = Image.open(CUR_DIR + '/Assets/Meteors/' + image_name).convert('RGBA')
            width, height = img.size
            new_height = int(height * PRM.METEOR_SCALING)
            new_width = int(width * PRM.METEOR_SCALING)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)

            m_sprite = Sprite(CUR_DIR + '/Assets/Meteors/' + image_name, scale=PRM.METEOR_SCALING)
            self.sprite_cache['meteor' + str(idx)] = m_sprite

            for i in range(360):
                id_str = 'meteor' + str(idx) + '-' + str(i)
                self.image_cache[id_str] = img.rotate(i, resample=Image.BICUBIC, expand=True)

        self.image_cache['background'] = Image.open(CUR_DIR + '/Assets/black.png').resize((600, 400))
        self.image_cache['background_1'] = Image.open(CUR_DIR + '/Assets/black.png').resize((600, 400))

        player_sprite = Player(CUR_DIR + '/Assets/player.png', scale=PRM.PLAYER_SCALING,
                               center_x=30, center_y=PRM.SCREEN_HEIGHT/2)
        self.sprite_cache['player'] = player_sprite

        player_img = Image.open(CUR_DIR + '/Assets/player.png').convert('RGBA')
        width, height = player_img.size
        self.p_width = int(width * PRM.PLAYER_SCALING)
        self.p_height = int(height * PRM.PLAYER_SCALING)
        self.image_cache['player'] = player_img.resize((self.p_width, self.p_height), Image.ANTIALIAS)

    def draw_and_check_collision(self, meteors):
        self.count += 1

        back = self.image_cache['background']
        back1 = self.image_cache['background_1']

        player_img = self.image_cache['player']
        player_sprite = self.sprite_cache['player']
        pwidth, pheight = player_img.size
        px = player_sprite.center_x
        py = PRM.SCREEN_HEIGHT - player_sprite.center_y
        player_pos = (int(px - (pwidth//2)), int(py - (pheight//2)))

        meteor_images = []
        meteor_positions = []

        collision = False

        for meteor in meteors:
            m_img = self.image_cache[meteor.get_name_angle()]
            m_sprite = self.sprite_cache[meteor.get_name()]

            m_sprite.center_x = meteor.center_x
            m_sprite.center_y = meteor.center_y
            m_sprite.angle = meteor.angle

            mwidth, mheight = m_img.size
            cx = meteor.center_x
            cy = PRM.SCREEN_HEIGHT - meteor.center_y
            mpos = (int(cx - (mwidth//2)), int(cy - (mheight//2)))

            meteor_positions.append(mpos)
            meteor_images.append(m_img)

            collision = collision or player_sprite.collides_with_sprite(m_sprite)

        back1.paste(back, (0, 0))
        back1.paste(player_img, player_pos, player_img)

        for i in range(len(meteors)):
            back1.paste(meteor_images[i], meteor_positions[i], meteor_images[i])

        if self.save_img:
            back1.save('generated_images/obs' + str(self.count).zfill(4) + '.png')

        bnh = back1.resize((PRM.OUTPUT_WIDTH, PRM.OUTPUT_HEIGHT)).convert('L')
        img_np = np.asarray(bnh, dtype=np.uint8)

        return img_np, collision


class MeteorI(Meteor):
    def __init__(self, img_no, rocket_x=30.0, rocket_y=PRM.SCREEN_HEIGHT / 2.0):
        super().__init__(None, PRM.METEOR_SCALING, rocket_x, rocket_y)
        self.img_no = img_no

    def get_name(self):
        return 'meteor' + str(self.img_no)

    def get_name_angle(self):
        tmp_angle = int(self.angle) % 360
        return 'meteor' + str(self.img_no) + '-' + str(tmp_angle)


# The game class. It can be used to generate state, reward, is_done for training
# (by setting train_mode in constructor to True and by calling reset > one_step(action)
# Also it can run a trained model and save the images of every frame in 'generated_images'
# folder (by setting model_path and save_img = True)
class MeteorImg:
    def __init__(self, model_path=None, train_mode=True, save_img=False):
        self.train_mode = train_mode
        self.mi = MeteorImages(save_img=save_img)
        self.skip = 0

        if not train_mode:
            assert model_path is not None
            self.device = torch.device("cuda")
            self.net = DuelingDQN(PRM.INPUT_SHAPE, n_actions=PRM.N_ACTIONS).to(self.device)
            self.net.load_state_dict(torch.load(model_path))

    def setup(self):
        self.player_list = []
        self.meteor_list = []

        self.spawn_frame = random.randrange(PRM.METEOR_SPAWN_FRAME_LOW, PRM.METEOR_SPAWN_FRAME_HIGH)
        self.last_meteor_frame = 0
        self.score = 0
        self.status = PRM.STATUS_RUNNING

        img_no = random.randint(0, 7)
        meteor = MeteorI(img_no)
        self.meteor_list.append(meteor)

        self.player_sprite = self.mi.sprite_cache['player']
        self.player_sprite.center_x = 30
        self.player_sprite.center_y = PRM.SCREEN_HEIGHT/2
        self.player_list.append(self.player_sprite)
        self.action = 8

        if not self.train_mode:
            self.buffer = np.zeros(PRM.INPUT_SHAPE, dtype=np.float)

    def sprites_update(self):
        self.player_sprite.update()

        for meteor in self.meteor_list:
            if meteor.center_x < -10 or meteor.center_y > PRM.SCREEN_HEIGHT + 10 or meteor.center_y < -10:
                self.meteor_list.remove(meteor)
                if self.status == PRM.STATUS_RUNNING or self.train_mode:
                    self.score += 1
            else:
                meteor.update()

        if self.last_meteor_frame > self.spawn_frame:
            self.last_meteor_frame = 0
            self.spawn_frame = random.randrange(PRM.METEOR_SPAWN_FRAME_LOW,
                                                PRM.METEOR_SPAWN_FRAME_HIGH)
            img_no = random.randint(0, 7)
            meteor = MeteorI(img_no, self.player_sprite.center_x, self.player_sprite.center_y)
            self.meteor_list.append(meteor)

        if PRM.PRINT_FPS:
            cur_time = time.time()
            if cur_time - self.time > 1:
                self.time = cur_time
                print(self.frames)
                self.frames = 0
            self.frames += 1

        self.last_meteor_frame += 1

    def rl_action(self, action):
        if action == 0:
            self.player_sprite.change_y = PRM.MOVEMENT_SPEED
        if action == 1:
            self.player_sprite.change_y = PRM.MOVEMENT_SPEED
            self.player_sprite.change_x = PRM.MOVEMENT_SPEED
        if action == 2:
            self.player_sprite.change_x = PRM.MOVEMENT_SPEED
        if action == 3:
            self.player_sprite.change_x = PRM.MOVEMENT_SPEED
            self.player_sprite.change_y = -PRM.MOVEMENT_SPEED
        if action == 4:
            self.player_sprite.change_y = -PRM.MOVEMENT_SPEED
        if action == 5:
            self.player_sprite.change_x = -PRM.MOVEMENT_SPEED
            self.player_sprite.change_y = -PRM.MOVEMENT_SPEED
        if action == 6:
            self.player_sprite.change_x = -PRM.MOVEMENT_SPEED
        if action == 7:
            self.player_sprite.change_x = -PRM.MOVEMENT_SPEED
            self.player_sprite.change_y = PRM.MOVEMENT_SPEED
        if action == 8:
            self.player_sprite.change_y = 0
            self.player_sprite.change_y = 0

    def torch_action(self):
        img_np, collision = self.mi.draw_and_check_collision(self.meteor_list)
        self.skip += 1
        if collision:
            self.reset()

        if self.skip == 3:
            self.skip = 0
            img_np = np.expand_dims(img_np, axis=0).astype(np.float32) / 255.0

            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = img_np

            with torch.no_grad():
                state_a = np.array([self.buffer], copy=False, dtype=np.float32)
                state_v = torch.tensor(state_a).to(self.device)
                q_vals_v = self.net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                self.action = int(act_v.item())

        self.rl_action(self.action)
        self.sprites_update()

    def one_step(self, action):
        score_prev = self.score
        self.rl_action(action)
        self.sprites_update()
        score_new = self.score

        img_np, collision = self.mi.draw_and_check_collision(self.meteor_list)

        reward = score_new - score_prev

        if collision:
            self.status = PRM.STATUS_END_ANIMATION
            reward = -100

        return img_np, reward, collision, None

    def reset(self):
        self.setup()
        img_np, collision = self.mi.draw_and_check_collision(self.meteor_list)
        return img_np

