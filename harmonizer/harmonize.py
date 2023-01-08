import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from harmonizer.src import model
import random

def compose_images(foreground, background_path):
    # foreground = Image.open(foreground_path)
    # foreground_alpha = np.array(foreground.getchannel(3))
    # assert np.any(foreground_alpha == 0), 'foreground needs to have some transparency: {}'.format(foreground_path)
    
    background = Image.open(background_path)
    background = background.convert('RGBA')
    
    # Rotate the foreground
    # angle_degrees = random.randint(0, 359)
    angle_degrees = 0
    foreground = foreground.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)
    
    # Scale the foreground
    # scale = random.random() * .5 + .5 # Pick something between .5 and 1 
    scale = 0.5
    new_size = (int(foreground.size[0] * scale), int(foreground.size[1] * scale))
    foreground = foreground.resize(new_size, resample=Image.BICUBIC)
    
    # Add any other transformations here...
    
    # Choose a random x,y position for the foreground
    max_xy_position = (background.size[0] - foreground.size[0], background.size[1] - foreground.size[1])
    assert max_xy_position[0] >= 0 and max_xy_position[1] >= 0, \
        'foreground {} is to big for the background {}'.format(foreground_path, background_path)
    paste_position = (random.randint(0, max_xy_position[0]), random.randint(0, max_xy_position[1]))
    
    # Create a new foreground image as large as the background and paste it on top
    new_foreground = Image.new('RGBA', background.size, color = (0, 0, 0, 0))
    new_foreground.paste(foreground, paste_position)
        
    # Extract the alpha channel from the foreground and paste it into a new image the size of the background
    alpha_mask = foreground.getchannel(3)
    new_alpha_mask = Image.new('L', background.size, color=0)
    new_alpha_mask.paste(alpha_mask, paste_position)
    composite = Image.composite(new_foreground, background, new_alpha_mask)
    
    # Grab the alpha pixels above a specified threshold
    alpha_threshold = 200
    mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
    hard_mask = Image.fromarray(np.uint8(mask_arr) * 255, 'L')
    
    # Get the smallest & largest non-zero values in each dimension and calculate the bounding box
    nz = np.nonzero(hard_mask)
    bbox = [np.min(nz[0]), np.min(nz[1]), np.max(nz[0]), np.max(nz[1])] 

    return composite, hard_mask, bbox

def placeImage(foreground, fg_mask, background):
    fg_img = Image.open(foreground)
    topX = -100
    topY = 140
    comp = Image.open(background)
    comp.paste(fg_img, (topX,topY), mask = fg_mask)

    return comp

def get_harmonized(comp, mask):

    # comp = Image.open(composite_img).convert('RGB')
    # mask = Image.open(mask_img).convert('1')
    if comp.size[0] != mask.size[0] or comp.size[1] != mask.size[1]:
        print('The size of the composite image and the mask are inconsistent')
    # convert to tensor
    comp = tf.to_tensor(comp)[None, ...]
    mask = tf.to_tensor(mask)[None, ...]
    # pre-defined arguments
    cuda = torch.cuda.is_available()
    
    
    harmonizer = model.Harmonizer()
    if cuda:
        harmonizer = harmonizer.cuda()
    harmonizer.load_state_dict(torch.load('harmonizer/pretrained/harmonizer.pth'), strict=True)
    if cuda:
        comp = comp.cuda()
        mask = mask.cuda()

    with torch.no_grad():
        arguments = harmonizer.predict_arguments(comp, mask)
        harmonized = harmonizer.restore_image(comp, mask, arguments)

    output_img = tf.to_pil_image(harmonized.squeeze())
    # output_img.save('harmonized.jpg')
    return output_img


def harmonize_and_display(img_name, bg_name, save_dir='harmonizer_ops'):
    global harmonizer
    #load foreground image and generate mask for area that will be merged
    fg_img = Image.open(img_name)
    # fg_img = cv2.resize(fg_img, dsize=(700, 1400), interpolation=cv2.INTER_CUBIC)
    # fg_mask = createMask(img_name)
    fg_mask = Image.fromarray(np.array(fg_img.convert('L'))<255)
    fg_mask = Image.fromarray(np.array(fg_img.convert('L'))>0)

    # fg_img = np.asarray(fg_img)
    # fg_img = torch.from_numpy(fg_img).type(torch.FloatTensor) 
    # fg_img = fg_img.cuda()

    #load background image and create composite by pasting the foreground
    comp = Image.open(bg_name)
    # comp = np.asarray(comp)
    # comp = torch.from_numpy(comp).type(torch.FloatTensor) 
    # comp = comp.cuda()
    #Currently, for testing purposes, the foreground is pasted at (200,0) coords
    #TO-DO: Algorithm to help with placement of foreground

    topX = -100
    topY = 140

    comp.paste(fg_img, (topX,topY), mask = fg_mask)

    #Generate mask for input into the harmonizer
    mask = Image.fromarray(np.array(comp.convert('L'))<0)
    mask.paste(fg_mask, (topX, topY), mask = fg_mask)

    # prepare inputs for argument prediction
    _comp = tf.to_tensor(comp)[None, ...]
    _mask = tf.to_tensor(mask)[None, ...]
    # _comp = _comp.cuda()
    # _mask = _mask.cuda()
    #     _image = tf.to_tensor(image)[None, ...]

    # predict arguments
    with torch.no_grad():
        arguments = harmonizer.predict_arguments(_comp, _mask)
        _harmonized = harmonizer.restore_image(_comp, _mask, arguments)

    # get output image and plot
    output_img = tf.to_pil_image(_harmonized.squeeze())

    return output_img
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # _, axes = plt.subplots(1, 2, figsize=(15,15))
    # axes[0].set_title('Composite Image')
    # axes[0].imshow(comp)
    # axes[1].set_title('Output Image')
    # axes[1].imshow(output_img)
    # plt.savefig(f'{save_dir}/model_op_{bg_name.split("/")[-1].split(".")[0]}_{img_name.split("/")[-1].split(".")[0]}.jpg', bbox_inches='tight', pad_inches=0.5)
    # plt.show