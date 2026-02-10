from data.loader import SimpleCubePPDataset

train_ds = SimpleCubePPDataset("train")
test_ds = SimpleCubePPDataset("test")

for epoch in range(10):
    for batch_images, batch_illum in train_ds.batches(32):
        pass

        print(batch_images.shape, batch_illum.shape)
