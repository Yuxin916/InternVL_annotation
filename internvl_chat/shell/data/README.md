## 📚 Prepare Habitat Training Data
1. Prepare **meta_path** at [habitat_hdt.json](habitat_hdt.json)
   - root is the root directory of the dataset
   - annotation is the path to the **_annotation file_**
   - data_augment indicates whether data augmentation is needed, 
   - repeat_time is the number of times the dataset is repeated, 
   - length is the number of samples in the dataset

2. Prepare the annotation format, can be four types:
   - Pure text data, 
   - Single-image data, 
   - Multi-image (interleaved) data, 
   - Video data
   - (We do not require all entries in a JSONL file to be of the same type, meaning your JSONL file can contain **different types of data**.)
   - See [official doc](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#meta-file) for more details