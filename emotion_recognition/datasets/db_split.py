import numpy as np

class SplittedDataset:
    """ like torch.utils.data.Subset()
        but:
            - take idxs in multiple convenient formats
            - repr(self) is improved

    split can be:
     - slice
     - tuple for cross-val: (split_num, total_splits) with 1 <= split_num <= total_splits
                            if split_num < 0: then all other splits are selected
    """
    def __init__(self, db, split, transform=None):
        self.db = db
        self.transform = transform
        if isinstance(split, slice):
            self.idxs = np.arange(len(db))[split]
            self.repr = f"[{split.start}:{split.end}:{split.step}]"

        elif isinstance(split, tuple):
            assert len(split) == 2
            split_num, total_splits = split
            assert 2 <= total_splits <= 10 and 1 <= abs(split_num) <= total_splits
            start = len(db) * (abs(split_num)-1) // total_splits
            end = len(db) * (abs(split_num)-0) // total_splits

            if split_num > 0:
                self.idxs = np.arange(start, end)
                self.repr = f"(validation split {split_num}/{total_splits})"
            else:
                self.idxs = np.r_[np.arange(0,start), np.arange(end, len(db))]
                self.repr = f"(train split {-split_num}/{total_splits})"

        elif isinstance(split, np.ndarray):
            assert split.dtype in (np.int32, np.int64)
            self.idxs = split
            self.repr = f"({len(self.idxs)}/{len(db)} indexed samples)"

        else:
            raise NotImplementedError(f"unknown split type {type(split)}")

    def __repr__(self):
        return f"Subset{self.repr} of " + repr(self.db)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        image, label = self.db[self.idxs[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# Split = SplittedDataset
# CrossVal = lambda split_num, total_splits, db: Split(db, (split_num, total_splits))
