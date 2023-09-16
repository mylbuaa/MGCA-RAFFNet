class LCU(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features,1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.GELU()
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
        #     nn.BatchNorm2d(hidden_features),
        #     nn.GELU()
        # )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
        #     nn.BatchNorm2d(hidden_features),
        #     nn.GELU()
        # )
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
        #     nn.BatchNorm2d(hidden_features),
        #     nn.GELU()
        # )
        # self.conv7 = nn.Sequential(
        #     nn.Conv2d(hidden_features, out_features, 1, bias=False),
        #     nn.GELU()
        # )
    
    def forward(self, x):
        x=torch.unsqueeze(x,1) 
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x=torch.squeeze(x,1) 
        # x = x + self.conv4(x)
        # x = self.conv5(x)
        # x = x + self.conv6(x)
        # x = self.conv7(x)
        return x
