import math
import arcade
import random
import os
import time
import numpy
from DQN_Dueling.model import DuelingDQN
import torch
from PIL import Image
import MeteorGame.parameters as PRM


class Meteor(arcade.Sprite):
    def __init__(self, image, scale, rocket_x=30.0, rocket_y=PRM.SCREEN_HEIGHT / 2.0):
        super().__init__(image, scale)
        self.acceleration_x = -random.random() * PRM.ACCELERATION_X_MULTIPLIER
        self.acceleration_y = (random.random() - 0.5) * PRM.ACCELERATION_Y_MULTIPLIER
        self.reset(rocket_x, rocket_y)

    def reset(self, rocket_x=30.0, rocket_y=PRM.SCREEN_HEIGHT / 2.0):
        self.angle = random.randrange(0, 360)

        rocket_x += random.randrange(-PRM.ROCKET_RANGE_LIMIT, PRM.ROCKET_RANGE_LIMIT)
        rocket_y += random.randrange(-PRM.ROCKET_RANGE_LIMIT, PRM.ROCKET_RANGE_LIMIT)

        self.center_x = PRM.SCREEN_WIDTH + PRM.METEOR_SPAWN_DIST_FROM_RIGHT_MARGIN
        self.center_y = random.randrange(50, PRM.SCREEN_HEIGHT - 50)

        self.change_x = random.randrange(PRM.HORIZONTAL_VELOCITY_LOW, PRM.HORIZONTAL_VELOCITY_HIGH)

        dx = rocket_x - self.center_x
        dy = rocket_y - self.center_y

        if dy < 0:
            self.acceleration_y *= -1

        a = self.acceleration_x
        b = 2 * self.change_x
        c = -2 * dx

        t1 = (-b + (math.sqrt(b ** 2 - 4 * a * c))) / (2 * a)
        t2 = (-b - (math.sqrt(b ** 2 - 4 * a * c))) / (2 * a)
        t = max(t1, t2)

        tmp = dy - (0.5 * self.acceleration_y * t ** 2)
        self.change_y = tmp / t

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y

        self.change_x += self.acceleration_x
        self.change_y += self.acceleration_y

        self.angle += PRM.ANGLE_CHANGE_SPEED


class Player(arcade.Sprite):
    def __init__(self, image, scale, center_x, center_y):
        super().__init__(image, scale)
        self.center_x = center_x
        self.center_y = center_y

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y

        if self.left < 0:
            self.left = 0
        elif self.right > PRM.SCREEN_WIDTH - 1:
            self.right = PRM.SCREEN_WIDTH - 1

        if self.bottom < 0:
            self.bottom = 0
        elif self.top > PRM.SCREEN_HEIGHT - 1:
            self.top = PRM.SCREEN_HEIGHT - 1


class Explosion(arcade.Sprite):
    def __init__(self, texture_list):
        super().__init__()
        self.current_texture = 0
        self.textures = texture_list

    def update(self):
        self.current_texture += 1
        if self.current_texture < len(self.textures):
            self.set_texture(self.current_texture)
        else:
            self.remove_from_sprite_lists()


class MeteorArcade(arcade.Window):
    def __init__(self, model_path=None, rl_mode=False, save_img=False):
        super().__init__(PRM.SCREEN_WIDTH, PRM.SCREEN_HEIGHT + PRM.SCREEN_SCORE_HEIGHT, PRM.SCREEN_TITLE)
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)
        self.rl_mode = rl_mode
        self.save_img = save_img
        self.count = 0

        self.model_path = model_path

        if model_path is not None:
            self.device = torch.device("cuda")
            self.net = DuelingDQN(PRM.INPUT_SHAPE, n_actions=PRM.N_ACTIONS).to(self.device)
            self.net.load_state_dict(torch.load(model_path))

    def setup(self):
        self.player_list = arcade.SpriteList()
        self.meteor_list = arcade.SpriteList()
        self.explosions_list = arcade.SpriteList()

        self.background = None

        self.spawn_frame = random.randrange(PRM.METEOR_SPAWN_FRAME_LOW, PRM.METEOR_SPAWN_FRAME_HIGH)
        self.last_meteor_frame = 0
        self.player_sprite = None
        self.score = 0
        self.skip = 0

        if self.rl_mode:
            self.status = PRM.STATUS_PAUSED
            self.set_update_rate(1.0 / PRM.UPDATE_RATE_RL)

        else:
            self.status = PRM.STATUS_RUNNING
            self.set_update_rate(1.0 / PRM.UPDATE_RATE)

        self.explosion_texture_list = []
        self.background = arcade.load_texture('Assets/black.png')
        file = 'Assets/Meteors/' + PRM.METEORS_IMAGES[random.randint(0, 7)]
        meteor = Meteor(file, PRM.METEOR_SCALING)
        self.meteor_list.append(meteor)

        self.player_sprite = Player("Assets/player.png", PRM.PLAYER_SCALING, 30, PRM.SCREEN_HEIGHT / 2)
        self.player_list.append(self.player_sprite)

        columns = 16
        count = 60
        sprite_width = 256
        sprite_height = 256
        file_name = ":resources:images/spritesheets/explosion.png"

        self.explosion_texture_list = arcade.load_spritesheet(file_name, sprite_width, sprite_height, columns, count)
        self.hit_sound = arcade.sound.load_sound(":resources:sounds/explosion2.wav")

        self.action = 8

        if self.model_path is not None:
            self.buffer = numpy.zeros(PRM.INPUT_SHAPE, dtype=numpy.float)

    # def on_draw(self):
    #     pass

    def meteors_draw(self):
        arcade.start_render()
        arcade.draw_lrwh_rectangle_textured(0, 0, PRM.SCREEN_WIDTH, PRM.SCREEN_HEIGHT, self.background)
        arcade.draw_text(f"Score: {self.score}", 260, 402, arcade.color.WHITE, 12)

        if self.status == PRM.STATUS_RUNNING or self.status == PRM.STATUS_END_ANIMATION:
            self.meteor_list.draw()
            self.player_list.draw()
            self.explosions_list.draw()

        elif self.status == PRM.STATUS_ENDED:
            output = "Game Over"
            arcade.draw_text(output, 100, 200, arcade.color.WHITE, 54)

    def on_update(self, delta_time):
        if self.status == PRM.STATUS_RUNNING or self.status == PRM.STATUS_END_ANIMATION:
            self.sprites_update()

        self.meteors_draw()

        if not self.rl_mode:
            self.save()

    def sprites_update(self):
        if self.status == PRM.STATUS_RUNNING and self.model_path is not None:
            self.torch_action()

        if self.status == PRM.STATUS_END_ANIMATION and len(self.explosions_list) == 0:
            self.status = PRM.STATUS_ENDED

        for meteor in self.meteor_list:
            if meteor.center_x < -10 or meteor.center_y > PRM.SCREEN_HEIGHT + 10 or meteor.center_y < -10:
                meteor.remove_from_sprite_lists()
                if self.status == PRM.STATUS_RUNNING or self.rl_mode:
                    self.score += 1
            else:
                meteor.update()

                collision_list = arcade.check_for_collision_with_list(meteor, self.player_list)

                if len(collision_list) > 0:
                    arcade.sound.play_sound(self.hit_sound)
                    explosion = Explosion(self.explosion_texture_list)

                    explosion.center_x = collision_list[0].center_x
                    explosion.center_y = collision_list[0].center_y

                    explosion.update()

                    self.explosions_list.append(explosion)
                    collision_list[0].remove_from_sprite_lists()
                    meteor.remove_from_sprite_lists()
                    self.status = PRM.STATUS_END_ANIMATION

        if self.last_meteor_frame > self.spawn_frame:
            self.last_meteor_frame = 0
            self.spawn_frame = random.randrange(PRM.METEOR_SPAWN_FRAME_LOW, PRM.METEOR_SPAWN_FRAME_HIGH)
            file = 'Assets/Meteors/' + PRM.METEORS_IMAGES[random.randint(0, 7)]
            meteor = Meteor(file, PRM.METEOR_SCALING, self.player_sprite.center_x, self.player_sprite.center_y)
            self.meteor_list.append(meteor)

        if PRM.PRINT_FPS:
            cur_time = time.time()
            if cur_time - self.time > 1:
                self.time = cur_time
                print(self.frames)
                self.frames = 0
            self.frames += 1

        self.last_meteor_frame += 1

        self.player_list.update()
        self.explosions_list.update()

    def on_key_press(self, key, modifiers):
        if not self.rl_mode:
            if key == arcade.key.UP:
                self.player_sprite.change_y = PRM.MOVEMENT_SPEED
            if key == arcade.key.DOWN:
                self.player_sprite.change_y = -PRM.MOVEMENT_SPEED
            if key == arcade.key.LEFT:
                self.player_sprite.change_x = -PRM.MOVEMENT_SPEED
            if key == arcade.key.RIGHT:
                self.player_sprite.change_x = PRM.MOVEMENT_SPEED

        if self.status == PRM.STATUS_ENDED and key == arcade.key.SPACE:
            self.setup()

    def on_key_release(self, key, modifiers):
        if not self.rl_mode:
            if key == arcade.key.UP or key == arcade.key.DOWN:
                self.player_sprite.change_y = 0
            elif key == arcade.key.LEFT or key == arcade.key.RIGHT:
                self.player_sprite.change_x = 0

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

    def save(self):
        if self.save_img:
            img_arcade = arcade.get_image(0, 0, PRM.SCREEN_WIDTH, PRM.SCREEN_HEIGHT + PRM.SCREEN_SCORE_HEIGHT)
            img = Image.fromarray(numpy.array(img_arcade))
            img.save('generated_images/obs' + str(self.count) + '.png')
            self.count += 1

    def reset(self):
        self.setup()
        self.meteors_draw()
        return self.image_to_torch()

    def image_to_torch(self):
        img_arcade = arcade.get_image(0, 0, PRM.SCREEN_WIDTH, PRM.SCREEN_HEIGHT)
        img = Image.fromarray(numpy.array(img_arcade))
        img = img.resize((PRM.OUTPUT_WIDTH, PRM.OUTPUT_HEIGHT)).convert('L')
        return numpy.array(img, dtype=numpy.float32) / 255.0

    def torch_action(self):
        self.skip += 1

        if self.skip == 3:
            self.skip = 0
            img = self.image_to_torch()
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = img
            with torch.no_grad():
                state_a = numpy.array([self.buffer], copy=False)
                state_v = torch.tensor(state_a).to(self.device, dtype=torch.float)
                q_vals_v = self.net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                self.action = int(act_v.item())
        self.rl_action(self.action)

    def one_step(self, action):
        self.save()
        score_prev = self.score
        self.rl_action(action)
        self.sprites_update()
        self.meteors_draw()
        score_new = self.score

        img = self.image_to_torch()

        is_done = self.status == PRM.STATUS_ENDED or self.status == PRM.STATUS_END_ANIMATION

        reward = score_new - score_prev

        if self.status == PRM.STATUS_ENDED or self.status == PRM.STATUS_END_ANIMATION:
            reward = -100

        return img, reward, is_done, None
