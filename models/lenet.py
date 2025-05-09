import torch.nn as nn

# class LeNet(nn.Module):
#     def __init__(self, channel=3, hideen=768, num_classes=10):
#         super(LeNet, self).__init__()
#         act = nn.Sigmoid
#         self.body = nn.Sequential(
#             nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
#             act(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(hideen, num_classes)
#         )
#
#     def forward(self, x):
#         out = self.body(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         # if return_interval:
#         #     return [out.clone()]
#         return out
class LeNet(nn.Module):
    def __init__(self, channel=3, hidden=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),  # 0
            act(),                                                             # 1
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),       # 2
            act(),                                                             # 3
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),       # 4
            act(),                                                             # 5
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)                                    # 6
        )

        # Combine all modules into a single list for flexible control
        self.splited_modules = list(self.body) + list(self.fc)
        self.length = len(self.splited_modules)

    def forward(self, x, starting_id=0, return_interval=False):
        if return_interval:
            res = []

        for i in range(starting_id, self.length):
            x = self.splited_modules[i](x)

            if i == self.length - 2:  # Before the Linear layer
                x = x.view(x.size(0), -1)

            if return_interval:
                res.append(x.clone())

        return res if return_interval else x


def lenet(channel=3, hidden=768, num_classes=10):
    return LeNet(channel=channel, hidden=hidden, num_classes=num_classes)

def param_name_to_module_id_lenet(name = 'depth'):
    if name.startswith('body.0'):
        return 0
    elif name.startswith('body.2'):
        return 1
    elif name.startswith('body.4'):
        return 2
    elif name.startswith('fc.0'):
        return 3
    else:
        raise NotImplementedError(f"Unknown parameter name: {name}")