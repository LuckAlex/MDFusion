import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from loss.id_loss import IDLoss
from PIL import Image
import io
from loss.L1_msssim import CombinedLoss
# from fe2 import CustomModel
# from fpn_gpt import CustomM
# from fe import CustomModel
# odel
from fe_test import CustomModel
import os
import argparse
import torch.multiprocessing as mp
import lmdb
# from fe3 import CustomModel
from loss.w_loss import WLoss



class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.length = self.get_length()

    def open_env(self):
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return env

    def get_length(self):
        with self.open_env() as env:
            with env.begin(write=False) as txn:
                length = int(txn.get("length".encode()))
        return length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.open_env() as env:
            with env.begin() as txn:
                img_key = f"image-{idx:08}"
                img_buffer = txn.get(img_key.encode())
        img = Image.open(io.BytesIO(img_buffer))  # Read the image using PIL

        if self.transform:
            img = self.transform(img)

        return img, 0  # Return 0 as a placeholder for labels, since labels are not used in your code

def data_loader(image_size, batch_size, dataset_root, rank, world_size):
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor()
    ])

    dataset = LMDBDataset(lmdb_path=dataset_root, transform=data_transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataset_loader = DataLoader(dataset,
                                batch_size=batch_size, sampler=sampler,
                                num_workers=4,
                                pin_memory=True,
                                drop_last=True)
    return dataset_loader


def load_model(save_name, model):
    model_data = torch.load(save_name, map_location='cpu')
    model_state_dict = model_data['model_state_dict']

    # Modify the keys in state_dict to match the keys in the multi-GPU model
    new_state_dict = {}
    for key, value in model_state_dict.items():
        new_key = 'module.' + key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    print("Model load success")



class FinalLosses(nn.Module):
    def __init__(self, ckpt, device, batch_size):
        super(FinalLosses, self).__init__()
        self.id_loss = IDLoss().eval()
        self.reco_loss = CombinedLoss()
        self.w_loss = WLoss(ckpt, device, batch_size)

    def forward(self, inputs, outputs):
        # torch_resize = transforms.Resize([112, 112])
        # loss_id = self.id_loss(torch_resize(inputs), torch_resize(outputs))
        loss_id = self.id_loss(inputs, outputs)
        loss_reco = self.reco_loss(inputs, outputs)

        loss_w_cos, loss_was = self.w_loss(inputs, outputs)
        loss = 0.95 * loss_reco + 0.01 * loss_id + 0.03 * loss_w_cos + 0.01 * loss_was
        # loss = 0.95 * loss_reco + 0.05 * loss_id
        return loss, loss_reco, loss_id, loss_w_cos, loss_was

def train(rank, world_size, device, image_size, batch_size, dataset_root, save_path, epochs):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    dataset_loader = data_loader(image_size, batch_size, dataset_root, rank, world_size)
    net = CustomModel().to(rank)
    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # optimizer = optim.Adam(net.parameters(), lr=0.0002)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    save_name = './checkpoint/checkpoint_9_epoch.pkl'
    load_model(save_name, net)
    # loss_function = FinalLosses().to(rank).eval()
    ckpt = './pretrained/e4e_ffhq_encode.pt'
    loss_function = FinalLosses(ckpt, rank, batch_size).to(rank)

    for epoch in range(epochs):
        running_loss = 0.0
        running_reco = 0.0
        running_id = 0.0
        running_w_cos = 0.0
        running_was = 0.0

        for i, data in enumerate(dataset_loader):
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)  # Get alpha from the model output
            loss, loss_reco, loss_id, loss_w_cos, loss_was = loss_function(inputs, outputs)
            loss.backward()
            optimizer.step()
            # print("reco=",loss_reco, flush=True)
            # print("id=",loss_id, flush=True)
            # print("w_cos=",loss_w_cos, flush=True)
            # print("was=",loss_was, flush=True)

            running_loss += loss.item()
            running_reco += loss_reco.item()
            running_id += loss_id.item()
            running_w_cos += loss_w_cos.item()
            running_was += loss_was.item()

        epoch_loss = running_loss / (i + 1)
        epoch_reco = running_reco / (i + 1)
        epoch_id = running_id / (i + 1)
        epoch_w_cos = running_w_cos / (i + 1)
        epoch_was = running_was / (i + 1)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Loss_reco: {epoch_reco:.4f}, loss_id: {epoch_id:.4f}, Loss_w_cos: {epoch_w_cos:.4f}, loss_was: {epoch_was:.4f}")
            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Loss_reco: {epoch_reco:.4f}, loss_id: {epoch_id:.4f}")

        if rank == 0 and (epoch + 1) % 5 == 0:
            checkpoint = {"model_state_dict": net.module.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = f"{save_path}/checkpoint_{epoch}_epoch.pkl"
            torch.save(checkpoint, path_checkpoint)

        torch.cuda.empty_cache()

    print('Finished Training')



def main():
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'

    image_size = 256
    batch_size = 8
    dataset_root = './ffhq.lmdb'
    save_path = './checkpoint'
    epochs = 1000
    mp.spawn(train,
             args=(world_size, "cuda", image_size, batch_size, dataset_root, save_path, epochs),
             nprocs=world_size,
             join=True)



if __name__ == '__main__':
    main()