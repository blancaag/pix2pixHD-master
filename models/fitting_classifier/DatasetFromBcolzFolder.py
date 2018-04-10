def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

# for k in range(5):
#     exec(f'cat_{k} = k*2')

# data_dir = 'data_multi_channel_transform'
# split = ['train', 'val'] # data_subdir
# data_labels = ['clean', 'deleted']
# data_size = (224, 224, 8)

class DatasetFromBcolzFolder(data.Dataset):
    
    def __init__(self, data_dir, data_transforms=None, label_transforms=None):
        
        super(DatasetFromBcolzFolder, self).__init__()
#         self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        
        self.data_dir = data_dir
#         for i in os.listdir(self.data_dir):
#             print(i)
        self.data_labels = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]
        
        print(self.data_labels)
            
#         self.target_size = target_size

        """create a numpy with all the images in that 'label' folder"""
        
        np_data = np.empty((0,) + data_size)
        np_labs = np.empty((0,))
        
        for i in self.data_labels:

            data = load_array(os.path.join(data_dir, i))
            labs = np.repeat(self.data_labels.index(i), data.shape[0])
                
            np_data = np.append(np_data, data, axis=0)
            np_labs = np.append(np_labs, labs, axis=0)
            
        print(np_data.shape, np_labs.shape)
        
        self.np_dataset = (np_data, np_labs)
        
        print(self.np_dataset[0].shape, self.np_dataset[1].shape)

        """transform to tensor dataset"""

        self.data = torch.from_numpy(self.np_dataset[0]).transpose(1, 3).float()
        self.labels = torch.from_numpy(self.np_dataset[1]).long()

#         self.dataset = torch.utils.data.TensorDataset(x, y) 
    
        print(self.data.shape, self.labels.shape)
            
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms

    def __getitem__(self, index):

        data = self.data[index]
        label = self.labels[index]
        
#         print(data.shape, label.shape)
        
        if self.data_transforms:
            for i in self.data_transforms:
                data = self.data_transforms[i](data)
            
        if self.label_transforms:
            for i in self.label_transforms:
                label = self.label_transforms[i](label)

        return data, label

    def __len__(self):
        
        return len(self.labels)