import pygame
import redis_io
import redis
import cv2
from vision.unet import UNetWrapper
from vision import median_path 
import numpy as np

# Home
PI_IP = '192.168.0.11'
# Noisebridge
# PI_IP = '10.21.1.214'
PORT=6379
display_width = 800
display_height = 600
unet = UNetWrapper()
unet.generate_model("/Users/tjmelanson/development/tensorflow_sandbox/training_best/cp-best.ckpt")

pygame.init()
clock = pygame.time.Clock()
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Mary controller')
r = redis.Redis(PI_IP, port=PORT, db=0)
font = pygame.font.Font('freesansbold.ttf', 24)

while True:
    events = pygame.event.get()
    control_cmd_key = 'redis_control'
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                print('LEFT')
                r.set(control_cmd_key, 'LEFT')
            if event.key == pygame.K_RIGHT:
                print('RIGHT')
                r.set(control_cmd_key, 'RIGHT')
            if event.key == pygame.K_UP:
                print('FWD')
                r.set(control_cmd_key, 'FWD')
            if event.key == pygame.K_DOWN:
                print('STOP')
                r.set(control_cmd_key, 'STOP')
    pygame.display.update()
    # TODO : split this into separate function
    img = redis_io.get_np_image_3d("img", r)
    img = bgr_to_rgb(img)
    cv2.imwrite("input.png", img)
    mask = unet.predict(img)
    cv2.imwrite("predict.png", img_overlay)
    img_overlay = img_overlay.transpose([1, 0, 2])
    median_path.draw_median_line(img_overlay, median_path.median_line(mask))
    surface = pygame.surfarray.make_surface(img_overlay).convert()
    gameDisplay.blit(surface, (0,0))
    clock.tick(30)


