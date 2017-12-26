#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging
import os

import imageio
from PIL import ImageDraw, Image, ImageFont

import lib

logging.basicConfig(level=logging.DEBUG)


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    # Reference Variables
    posts = [['one of the biggest scams is believing', 'one of the biggest scams is believing to suffer'],
             ['dogs are really just people that should', 'dogs are really just people that should live to kill'],
             ['smart phones are today s version of the',
              'smart phones are today s version of the friend to the millions']
             ]

    # Iterate through posts
    for (post_seed, generated_post) in posts:
        logging.info('Working generated post: {}'.format(generated_post))

        # Feed each seed and respose to helper method
        create_viz(post_seed, generated_post)

    pass


def create_viz(post_seed, generated_post):
    # Reference variables
    post_steps = list()
    font_size = 15
    image_height = font_size * 3
    image_width = int(len(generated_post) * font_size * .6)
    cell_font = ImageFont.truetype('../resources/Vera.ttf', size=15)

    # TODO Set up chunks (new word, each individual character)
    post_steps.append(post_seed)
    for char in generated_post[len(post_seed):]:
        post_steps.append(char)

    logging.info('Post steps: {}'.format(post_steps))

    image_paths = list()

    # TODO Iterage through each chunk
    for step_index in range(1, len(post_steps) + 1):
        step_text = ''.join(post_steps[:step_index])

        # Create image of start to current chunk
        cell_image = Image.new('RGB', (image_width, image_height), color='white')
        d = ImageDraw.Draw(cell_image)

        # Draw text, half opacity

        d.text((10, 10), step_text, font=cell_font, fill=(128, 128, 128))

        # Draw seed, full opacity
        d.text((10, 10), post_seed, font=cell_font, fill=(16, 16, 16))

        cell_save_path = os.path.join(lib.get_conf('viz_intermediate_path'), post_seed + str(step_index) + '.tiff')
        cell_image.save(cell_save_path)
        image_paths.append(cell_save_path)

    # Create gif from all file paths
    images = list()
    gif_path = os.path.join(lib.get_conf('viz_gif_path'), post_seed + '.gif')
    for filename in image_paths:
        images.append(imageio.imread(filename))

    # Add buffer for first and last


    images = images[:1] * 6 + images + images[-1:] * 6
    imageio.mimsave(gif_path, images)


# Main section
if __name__ == '__main__':
    main()
