""" Utilities to create Sport Balls dataset

Author: Magda Gregorova
Created: 2024-06-16
"""

# import os
# import csv
# import torch
# from PIL import Image, ImageOps


# def _sports_balls_generator(train=False, canvas_size=64, object_size=34, n_objects=1, random_cls=False):
#     """Generate labeled data of sports balls

#     Args:
#         train: boolean to switch manual seed for random between train and test
#         canvas_size: integer size of the complete background image
#         object_size: integer starting size of the objects to be placed on canvas
#         n_objects: integer number of objects to be placed on canvas
#     """
#     baseball = Image.open("data/objects/baseball.png")
#     basketball = Image.open("data/objects/basketball.png")
#     volleyball = Image.open("data/objects/volleyball.png")
#     soccerball = Image.open("data/objects/soccerball.png")
#     sportballs = [baseball, basketball, volleyball, soccerball]
#     rnd_gen = torch.Generator().manual_seed(42) if train else torch.Generator().manual_seed(24)
#     rnd_state = rnd_gen.initial_seed()

#     while True:
#         background = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))

#         class_idx = torch.randint(0, len(sportballs), (n_objects,), generator=rnd_gen)
#         rotation = torch.randint(0, 360, (n_objects,), generator=rnd_gen)
#         size = (torch.randint(5, 10, (n_objects,), generator=rnd_gen) * object_size / 10.).long()
#         pos_idx = torch.zeros(n_objects, 2)  # top-left corner of object
#         for i in range(n_objects):
#             pos_idx[i, :] = torch.rand((1, 2), generator=rnd_gen)*(canvas_size - size[i].item())

#         for cls, pos, rot, s in zip(class_idx, pos_idx.long(), rotation, size):
#             obj = sportballs[cls].rotate(rot).resize((s, s))
#             background.paste(obj, pos.tolist(), obj)
#         if random_cls:  # for random colormap
#             if rot.item() % 3 == 0:
#                 background = ImageOps.invert(background.convert('RGB')) # inverted colormap
#             elif rot.item() % 3 == 1:
#                 background.convert('L')  # grayscale colormap

#         # return the class(es) as label
#         yield background, class_idx, rnd_state


# def store_sports_balls(length, train=False, path='/home/magda/Data/SportBalls', **kwargs):
#     """Store sport balls data to a folder

#     Args:
#         length: integer number of examples
#         train: boolean if train or test data
#         path: string path to data storage
#     """
#     sb = _sports_balls_generator(train, kwargs['canvas_size'], kwargs['object_size'], kwargs['n_objects'], kwargs['random_cls'])
#     str_len = len(str(length))
#     path = path + ('/Train' if train else '/Test')

#     if not os.path.exists(path):
#         os.makedirs(path)
#         print(f'Folder {path} created.')

#     print(f'Will generate {length} images into {path}.')
#     labels_path = path + '/labels.csv'
#     pred_path = path + '/preds.csv'
#     with open(labels_path, 'w', newline='') as lab_file: 
#         lab_writer = csv.writer(lab_file, delimiter=',')
#         for i in range(length):
#             img, label, r_state = next(sb)
#             if i == 0:
#                 rnd_state = r_state
#             img_name = 'img_' + str(i).zfill(str_len)
#             lab_writer.writerow([img_name] + label.tolist())
#             img_name = path + '/' + img_name +  '.png'
#             img.save(img_name)
#             if i % 500 == 0:
#                 print(f'{i} images done and {length-i} still to do.')



import os
import csv
import torch
from PIL import Image, ImageOps

def _sports_balls_generator(train=False, canvas_size=64, object_size=34, n_objects=1, random_cls=False):
    """Generate labeled data of sports balls

    Args:
        train: boolean to switch manual seed for random between train and test
        canvas_size: integer size of the complete background image
        object_size: integer starting size of the objects to be placed on canvas
        n_objects: integer number of objects to be placed on canvas
    """
    baseball = Image.open("data/objects/baseball.png")
    basketball = Image.open("data/objects/basketball.png")
    volleyball = Image.open("data/objects/volleyball.png")
    soccerball = Image.open("data/objects/soccerball.png")
    sportballs = [baseball, basketball, volleyball, soccerball]
    
    # Define colors for different ball types in the mask (R, G, B)
    mask_colors = {
        0: (255, 0, 0),   # Baseball - Red
        1: (0, 255, 0),   # Basketball - Green
        2: (0, 0, 255),   # Volleyball - Blue
        3: (255, 255, 0)  # Soccerball - Yellow
    }
    
    rnd_gen = torch.Generator().manual_seed(42) if train else torch.Generator().manual_seed(24)
    rnd_state = rnd_gen.initial_seed()

    while True:
        background = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
        mask = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))  # Colored mask

        class_idx = torch.randint(0, len(sportballs), (n_objects,), generator=rnd_gen)
        rotation = torch.randint(0, 360, (n_objects,), generator=rnd_gen)
        size = (torch.randint(5, 10, (n_objects,), generator=rnd_gen) * object_size / 10.).long()
        pos_idx = torch.zeros(n_objects, 2)  # top-left corner of object
        for i in range(n_objects):
            pos_idx[i, :] = torch.rand((1, 2), generator=rnd_gen) * (canvas_size - size[i].item())

        for cls, pos, rot, s in zip(class_idx, pos_idx.long(), rotation, size):
            obj = sportballs[cls].rotate(rot).resize((s, s))
            background.paste(obj, pos.tolist(), obj)
            
            # Create the mask using the actual shape of the object
            mask_obj = Image.new("RGBA", obj.size, mask_colors[cls.item()])  # Use red, green, and blue for different ball types
            mask.paste(mask_obj, pos.tolist(), obj)
            
        if random_cls:  # for random colormap
            if rot.item() % 3 == 0:
                background = ImageOps.invert(background.convert('RGB'))  # inverted colormap
            elif rot.item() % 3 == 1:
                background.convert('L')  # grayscale colormap

        # return the class(es) as label
        yield background.convert('RGB'), mask.convert('RGB'), class_idx, rnd_state

def store_sports_balls(length, train=False, path='/home/magda/Data/SportBalls', **kwargs):
    """Store sport balls data to a folder

    Args:
        length: integer number of examples
        train: boolean if train or test data
        path: string path to data storage
    """
    sb = _sports_balls_generator(train, kwargs['canvas_size'], kwargs['object_size'], kwargs['n_objects'], kwargs['random_cls'])
    str_len = len(str(length))
    path = path + ('/Train' if train else '/Test')

    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Folder {path} created.')

    print(f'Will generate {length} images into {path}.')
    labels_path = os.path.join(path, 'labels.csv')
    with open(labels_path, 'w', newline='') as lab_file:
        lab_writer = csv.writer(lab_file, delimiter=',')
        for i in range(length):
            img, mask, label, r_state = next(sb)
            if i == 0:
                rnd_state = r_state
            img_name = 'img_' + str(i).zfill(str_len)
            lab_writer.writerow([img_name] + label.tolist())
            img_name_with_extension = img_name + '.png'
            img_path = os.path.join(path, img_name_with_extension)
            mask_path = os.path.join(path, 'masks', img_name_with_extension)
            
            # Ensure the mask directory exists
            os.makedirs(os.path.join(path, 'masks'), exist_ok=True)
            
            img.save(img_path)
            mask.save(mask_path)
            if i % 500 == 0:
                print(f'{i} images done and {length-i} still to do.')

# Example usage
# store_sports_balls(length=10000, train=True, path='path/to/data', canvas_size=64, object_size=34, n_objects=1, random_cls=False)
