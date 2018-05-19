# from nyu.loader import load_data, get_data_path, get_names
# data = load_data(get_data_path('v2'))
# print(get_names(data))
from nyu import NyuLoader, label_mapper
import matplotlib.pyplot as plt

super_cats = (
    ('floor', 'floor mat'),
    (
      'classroom board',
      'blinds',
      'window cover',
      'door frame',
      'door curtain',
      'wardrobe',
      'garage door',
      'cabinet',
      'wall',
      'wall decoration',
      'wall stand',
      'dishwasher',
      'door',
      'projector screen',
      'mirror',
      'mailshelf',
      'bookshelf',
      'storage shelvesbooks',
      'refridgerator',
    ),
    (
        'ceiling',
        # 'roof',  # no roof in original label names...
    ),
    ('table', 'desk', 'bed', 'coffee table', 'mattress', 'sofa')
)

super_cat_labels = ('ground', 'vertical', 'celiing', 'furnitures', 'objects')

with NyuLoader('v2') as loader:
    mapper = label_mapper(loader.label_names, super_cats)
    for i in range(len(loader)):
        example = loader.get_example(i)
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
        ax0.imshow(example.image)
        ax1.imshow(example.depth)
        ax2.imshow(example.labels)
        ax3.imshow(mapper(example.labels))
        plt.show()
