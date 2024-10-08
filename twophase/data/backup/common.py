import logging
from detectron2.data.common import MapDataset, AspectRatioGroupedDataset


class MapDatasetTwoCrop(MapDataset):
    """
    Map a function over the elements in a dataset.
    This customized MapDataset transforms an image with two augmentations
    as two inputs (queue and key).
    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class AspectRatioGroupedDatasetTwoCrop(AspectRatioGroupedDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.
    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        self._buckets_key = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            # d is a tuple with len = 2
            # It's two images (same size) from the same image instance
            w, h = d[0]["width"], d[0]["height"]
            bucket_id = 0 if w > h else 1

            # bucket = bucket for normal images
            bucket = self._buckets[bucket_id]
            bucket.append(d[0])

            # buckets_key = bucket for augmented images
            buckets_key = self._buckets_key[bucket_id]
            buckets_key.append(d[1])
            if len(bucket) == self.batch_size:
                yield (bucket[:], buckets_key[:])
                del bucket[:]
                del buckets_key[:]


# TODO: change or write a new class to accomodate more than 2
class AspectRatioGroupedSemiSupDatasetTwoCrop(AspectRatioGroupedDataset):
    """
    Extended class documentation...
    """

    def __init__(self, dataset_list, batch_size_list):
        """
        Method documentation...
        """
        self.dataset_list = dataset_list
        self.batch_size_list = batch_size_list

        self._buckets = [[[] for _ in range(2)] for _ in dataset_list]
        self._buckets_key = [[[] for _ in range(2)] for _ in dataset_list]

    def __iter__(self):
        buckets = [[[] for _ in range(2)] for _ in self.dataset_list]

        for data in zip(*self.dataset_list):
            for d, bucket, batch_size in zip(data, buckets, self.batch_size_list):
                if len(bucket[0]) != batch_size:
                    w, h = d["width"], d["height"]
                    bucket_id = 0 if w > h else 1
                    bucket[bucket_id].append(d)
                    
            if all(len(bucket[0]) == batch_size for bucket, batch_size in zip(buckets, self.batch_size_list)):
                yield tuple(bucket[0][:] for bucket in buckets)
                for bucket in buckets:
                    del bucket[0][:]


class AT_AspectRatioGroupedSemiSupDatasetTwoCrop(AspectRatioGroupedDataset):
    """
    Extended class documentation...
    """

    def __init__(self, dataset_list, batch_size_list):
        """
        Args:
            dataset_list: a list of iterable generators. Each dataset should produce
                          tuples where the first element is a dict representing data with strong augmentation
                          and the second is data with weak augmentation.
                          Each element in the tuple should be a dict with "width" and "height" keys.
            batch_size_list: a list of batch sizes for each dataset in dataset_list.
        """
        self.dataset_list = dataset_list
        self.batch_size_list = batch_size_list

        # The _buckets will store the strong augmentation data, 
        # while _buckets_key will store the weak augmentation data
        self._buckets = [[[] for _ in range(2)] for _ in dataset_list]
        self._buckets_key = [[[] for _ in range(2)] for _ in dataset_list]

    def __iter__(self):
        # Each dataset will have a bucket for strong augmentations and weak augmentations
        buckets = [[[] for _ in range(2)] for _ in self.dataset_list]
        buckets_key = [[[] for _ in range(2)] for _ in self.dataset_list]

        for data in zip(*self.dataset_list):
            for d, (bucket, bucket_key), batch_size in zip(data, zip(buckets, buckets_key), self.batch_size_list):
                # Data from strong augmentations
                if len(bucket[0]) != batch_size:
                    w, h = d[0]["width"], d[0]["height"]
                    bucket_id = 0 if w > h else 1
                    bucket[bucket_id].append(d[0])
                    
                # Data from weak augmentations
                if len(bucket_key[0]) != batch_size:
                    w, h = d[1]["width"], d[1]["height"]
                    bucket_id_key = 0 if w > h else 1
                    bucket_key[bucket_id_key].append(d[1])
                    
            if all(len(bucket[0]) == batch_size for bucket, batch_size in zip(buckets, self.batch_size_list)):
                yield tuple(bucket[0][:] + bucket_key[0][:] for bucket, bucket_key in zip(buckets, buckets_key))
                for bucket, bucket_key in zip(buckets, buckets_key):
                    del bucket[0][:]
                    del bucket_key[0][:]


# class AspectRatioGroupedSemiSupDatasetTwoCrop(AspectRatioGroupedDataset):
#     """
#     Extended class documentation...
#     """

#     def __init__(self, dataset, batch_size):
#         """
#         Method documentation...
#         """

#         self.label_dataset, self.unlabel_dataset, self.unlabel_dep_dataset = dataset
#         self.batch_size_label = batch_size[0]
#         self.batch_size_unlabel = batch_size[1]
#         self.batch_size_unlabel_dep = batch_size[2]  # I'm assuming you have a separate batch size for this

#         self._label_buckets = [[] for _ in range(2)]
#         self._label_buckets_key = [[] for _ in range(2)]
#         self._unlabel_buckets = [[] for _ in range(2)]
#         self._unlabel_buckets_key = [[] for _ in range(2)]
#         self._unlabel_dep_buckets = [[] for _ in range(2)]  # new bucket for unlabel_dep_dataset
#         self._unlabel_dep_buckets_key = [[] for _ in range(2)]  # new bucket key for unlabel_dep_dataset

#     def __iter__(self):
#         label_bucket, unlabel_bucket, unlabel_dep_bucket = [], [], []  # added unlabel_dep_bucket
#         for d_label, d_unlabel, d_unlabel_dep in zip(self.label_dataset, self.unlabel_dataset, self.unlabel_dep_dataset):  # added d_unlabel_dep

#             # existing handling for label bucket and unlabel bucket

#             if len(label_bucket) != self.batch_size_label:
#                 w, h = d_label["width"], d_label["height"]
#                 label_bucket_id = 0 if w > h else 1
#                 label_bucket = self._label_buckets[label_bucket_id]
#                 label_bucket.append(d_label)

#             if len(unlabel_bucket) != self.batch_size_unlabel:
#                 w, h = d_unlabel["width"], d_unlabel["height"]
#                 unlabel_bucket_id = 0 if w > h else 1
#                 unlabel_bucket = self._unlabel_buckets[unlabel_bucket_id]
#                 unlabel_bucket.append(d_unlabel)
            
#             # new handling for unlabel_dep_bucket
#             if len(unlabel_dep_bucket) != self.batch_size_unlabel_dep:
#                 w, h = d_unlabel_dep["width"], d_unlabel_dep["height"]
#                 unlabel_dep_bucket_id = 0 if w > h else 1
#                 unlabel_dep_bucket = self._unlabel_dep_buckets[unlabel_dep_bucket_id]
#                 unlabel_dep_bucket.append(d_unlabel_dep)

#             # checking if all buckets are full before yielding
#             if (
#                 len(label_bucket) == self.batch_size_label
#                 and len(unlabel_bucket) == self.batch_size_unlabel
#                 and len(unlabel_dep_bucket) == self.batch_size_unlabel_dep  # added this line
#             ):
#                 yield (
#                     label_bucket[:],
#                     unlabel_bucket[:],
#                     unlabel_dep_bucket[:],  # added this line
#                 )
#                 del label_bucket[:]
#                 del unlabel_bucket[:]
#                 del unlabel_dep_bucket[:]  # added this line