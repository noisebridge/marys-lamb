import pygame
import redis_io
import redis
import cv2
import numpy as np

# Home
PI_IP = '192.168.0.11'
# Noisebridge
# PI_IP = '10.21.1.214'
PORT=6379
display_width = 800
display_height = 600

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
    img = img[:, :, ::-1]
    img = img.transpose([1, 0, 2])
    cv2.imwrite("input.png", img)
    #mask = unet.predict(img)
    #img_overlay = cv2.resize(img, (128, 128))
    #img_overlay[:,:,1] = img_overlay[:,:,1]/2 + np.maximum(img_overlay[:,:,1], mask)/2
    #cv2.imwrite("predict.png", img_overlay)
    #median_path.draw_median_line(img_overlay, median_path.median_line(mask))

    surface = pygame.surfarray.make_surface(img).convert()
    gameDisplay.blit(surface, (0,0))
    clock.tick(30)


