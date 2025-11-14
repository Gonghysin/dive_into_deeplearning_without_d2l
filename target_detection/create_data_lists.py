from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(
        voc07_path='/root/autodl-tmp/dive_into_deeplearning_without_d2l/data/VOCdevkit/VOC2007',
        voc12_path='/root/autodl-tmp/dive_into_deeplearning_without_d2l/data/VOCdevkit/VOC2012',
        output_folder='/root/autodl-tmp/dive_into_deeplearning_without_d2l/data/VOCdevkit/',
    )
    
    print("Data lists created successfully.")