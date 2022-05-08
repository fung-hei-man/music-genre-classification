import random

INPUT_DIR = '/input'
CATEGORY = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def get_random_sample():
    rand = random.randint(0, 49)
    return random.choice(CATEGORY), f'000{rand}' if rand > 9 else f'0000{rand}'


def gen_file_name(genre, idx):
    return f'{INPUT_DIR}/{genre}.{idx}.wav'
