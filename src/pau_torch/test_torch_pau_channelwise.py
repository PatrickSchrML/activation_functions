import torch
from src.pau_torch.pade_activation_unit import PAU_conv

inp = torch.ones([1, 2, 8, 8], dtype=torch.float32)
#inp = inp.cuda()

pau = PAU_conv(channels=2, cuda=False)

for name, p in pau.named_parameters():
    if '0.center' in name:
        pass
    if '1.center' in name:
        p.data = torch.ones(p.shape)

for name, p in pau.named_parameters():
    if '.center' in name:
        print(name, p)

out = pau(inp)

print(out)