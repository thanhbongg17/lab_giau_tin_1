import sys
import torch
import torch.nn.utils.prune as prune
import torch.optim as optim
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    # — Auto-insert --dataroot nếu chưa có —
    if '--dataroot' not in sys.argv:
        sys.argv.extend(['--dataroot', './datasets/geo_wm'])
        print("[Info] --dataroot not provided, auto-using './datasets/geo_wm'")

    # 1. Parse toàn bộ options
    opt = TrainOptions().parse()

    # 2. Ép chạy ở phase=train và không huấn luyện lại từ đầu
    opt.phase = 'train'
    opt.epoch_count = 1
    opt.n_epochs = 0
    opt.n_epochs_decay = 0

    # 3. Tạo và setup model
    model = create_model(opt)
    model.setup(opt)

    # 4. Prune 20% trọng số Conv2d cho netG_A & netG_B
    for net in (model.netG_A, model.netG_B):
        for module in net.modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.2)

    # 5. Fine-tune netG_A trên dữ liệu sạch
    optimizer = optim.Adam(model.netG_A.parameters(), lr=opt.lr)
    criterion = torch.nn.L1Loss()
    dataset   = create_dataset(opt)

    model.netG_A.train()
    for epoch in range(1, 4):  # fine-tune 3 epochs
        total_loss = 0.0
        for i, data in enumerate(dataset):
            # set_input gán model.real_A, model.real_B
            model.set_input(data)

            real_A = model.real_A          # đúng API
            fake_B = model.netG_A(real_A)  # embed → output B
            rec_A  = model.netG_B(fake_B)  # reconstruct A

            loss = criterion(rec_A, real_A)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(dataset)
        print(f"Epoch {epoch} fine-tune loss: {avg:.4f}")

    # 6. Lưu mô hình đã clean
    clean_path = 'clean_model.pth'
    torch.save(model.netG_A.state_dict(), clean_path)
    print("Fine-tuning completed")
    print(f"Pruned & fine-tuned model saved to {clean_path}")
