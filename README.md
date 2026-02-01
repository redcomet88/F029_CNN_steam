# F029_CNN_steam CNN游戏推荐大数据系统vue+flask|python实现，使用CNN算法训练模型进行游戏推荐、还有游戏评论区讨论区功能

> 完整项目收费，可联系QQ: 81040295 微信: mmdsj186011 注明从github来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775
> 

关注B站，有好处！
编号:  **F029** CNN
## 视频

[video(video-f7uyWJ1k-1760594132972)(type-bilibili)(url-https://player.bilibili.com/player.html?aid=944050264)(image-https://i-blog.csdnimg.cn/img_convert/76afff6edc5c56699dcc964fc2c729ca.jpeg)(title-超帅vue+python游戏推荐大数据系统源码|协同过滤|可视化|KNN算法|推荐算法|Flask框架|MySQL|沙箱支付|短信接口|OCR识别|爬虫)]

## 1 系统简介
系统简介：本系统是一个基于Vue+Flask构建的游戏推荐与数据分析平台。其核心功能围绕用户管理、游戏推荐、社区互动和数据可视化展开。主要包括：个性化的登录与注册界面，支持背景视频播放；智能的游戏推荐系统，采用UserCF、ItemCF和CNN深度学习算法，提供多样化的推荐方式；丰富的游戏库和搜索功能，方便用户查找感兴趣的游戏；多维度的数据分析模块，包括游戏评价、类型分布和词云分析等可视化呈现；热闹的游戏讨论区，供用户交流游戏心得与推荐；以及完善的个人设置功能，支持用户信息和头像管理。系统还整合了数据爬虫模块，定期从Steam平台抓取最新游戏数据，确保内容的时效性和准确性。通过AI推荐算法和社区化功能，系统为用户提供了智能化、社交化的游戏体验。
## 2 功能设计
该系统采用典型的B/S（浏览器/服务器）架构模式。用户通过浏览器访问Vue前端界面，前端由HTML、CSS、JavaScript以及Vue.js生态系统中的Vuex（用于状态管理）、Vue Router（用于路由导航）和ECharts（用于数据可视化）等组件构建。前端通过API请求与Flask后端进行数据交互，Flask后端则负责业务逻辑处理，并利用SQLAlchemy等ORM工具与MySQL数据库进行数据存储。系统还部署了数据爬虫模块，用于从Steam平台抓取游戏数据并导入数据库，为平台提供数据支持。通过AI推荐算法，系统能够根据用户行为和游戏特征进行个性化推荐，为用户带来智能化的游戏推荐体验。
### 2.1系统架构图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b4dcb3955a4049949e232f1c9b682187.png)
### 2.2 功能模块图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a68f41ab106546658ba80a44de22cc7c.png)
## 3 功能展示
### 3.1 登录 & 注册
登录注册做的是一个可以切换的登录注册界面，点击去登录后者去注册可以切换，背景是一个视频，循环播放。
登录需要验证用户名和密码是否正确，如果**不正确会有错误提示**。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4311ebcb8d214134b4bafc5051b38603.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/200099b3c5674742a0e7f50c67818050.png)
注册需要**验证用户名是否存在**，如果错误会有提示。
### 3.2 主页
主页的布局采用了左侧是菜单，右侧是操作面板的布局方法，右侧的上方还有用户的头像和退出按钮，如果是新注册用户，没有头像，这边则不显示，需要在个人设置中上传了头像之后就会显示。
### 3.3 推荐算法
采用三种推荐算法，CNN推荐和UserCF、ItemCF进行游戏推荐，以卡片形式展示游戏内容。
CNN训练模型后推荐：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fa7f1f60b1be40869973c49a2f113f7a.png)
UserCF推荐：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/82f028cbf5ae4552854efa721a3bc323.png)
ItemCF推荐：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0a91886606a94ffda020b4826527ac65.png)
展示游戏视频：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fb15bc61ecc3452fbccdc321ced673a4.png)
### 3.4 游戏的评论区功能
查看单个游戏的评论区：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1fde2463dd7146719b8c46c30f96c926.png)
添加评论：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c2df29955b0441b93658e28f9598b87.png)
### 3.5 游戏库
可以在游戏库中搜索游戏，寻找自己感兴趣的游戏信息：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d9230926a48d484f8e28ad5bc6ef2e32.png)
搜索游戏：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6e6d406dbdff4a5a84d10d2137ba3f85.png)
### 3.6 数据分析
这边数据分析也是实现了多种不同的图形，比如
游戏评价：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/21079751e13a4e039d8982277943ba5a.png)
好评、差评
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ef8a254075224e9a8423f8e0e3d50911.png)
游戏类型散点图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f6b49cdf9d794932926bc32edfe749a8.png)
游戏介绍的词云分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c044a220b0bd4f6d8514abdd9fdf7847.png)
### 3.7 个人设置
个人设置方面包含了用户信息修改、密码修改功能。
用户信息修改中可以上传头像，完成用户的头像个性化设置，也可以修改用户其他信息。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f59e22973f1b4d39b57ec757aad72f27.png)
修改密码需要输入用户旧密码和新密码，验证旧密码成功后，就可以完成密码修改。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4bf27ead4c0d417dac9cef27da18ce06.png)
### 3.8 数据爬虫
可以爬取steam网站更新数据：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2e0c12bde8334adabc22db2b7d9f95fc.png)
## 4程序代码
### 4.1 代码说明
代码介绍：该算法利用卷积神经网络（CNN）从用户的历史评分和游戏特征中学习，生成个性化的游戏推荐列表。通过将用户评分和游戏特征转换为矩阵输入，CNN能够自动提取深层次的特征关系，以提高推荐的准确性和多样性。
### 4.2 流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d29cf7c7e066423c8fbaf6056eafbda1.png)

### 4.3 代码实例
```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# 数据加载与预处理
def load_data(file_path):
    data = pd.read_csv(file_path)
    user_dict = defaultdict(list)
    game_features = {}
    for _, row in data.iterrows():
        user = row['user_id']
        game = row['game_id']
        rating = row['rating']
        features = row['features']  # 假设有游戏特征列
        if game not in game_features:
            game_features[game] = features
        user_dict[user].append((game, rating))
    return user_dict, game_features

# 定义数据集类
class GameDataset(Dataset):
    def __init__(self, user_dict, game_features, game_id_mapping, user_id_mapping, seq_length=10):
        self.user_dict = user_dict
        self.game_features = game_features
        self.game_id_mapping = game_id_mapping
        self.user_id_mapping = user_id_mapping
        self.seq_length = seq_length
        self.data = self._prepare_data()
    
    def _prepare_data(self):
        data = []
        for user, interactions in self.user_dict.items():
            user_id = self.user_id_mapping[user]
            seq = []
            for game, rating in interactions:
                game_id = self.game_id_mapping[game]
                feature_vector = np.array(self.game_features[game]).astype(float)
                seq.append((game_id, feature_vector, rating))
            # 创建序列数据
            for i in range(len(seq) - self.seq_length):
                input_seq = seq[i:i + self.seq_length]
                input_features = [f for (g, f, r) in input_seq]
                target_rating = seq[i + self.seq_length][2]
                data.append((user_id, np.array(input_features), target_rating))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_id, features, rating = self.data[idx]
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'features': torch.tensor(features, dtype=torch.float),
            'rating': torch.tensor(rating, dtype=torch.float)
        }

# CNN模型定义
class CNNRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(CNNRecommender, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.fc1 = nn.Linear(30 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 30 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练模型
def train_model(model, device, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            user_ids = batch['user_id'].to(device)
            features = batch['features'].to(device)
            ratings = batch['rating'].to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 生成推荐
def generate_recommendations(model, device, user_id, game_features, game_id_mapping, k=5):
    model.eval()
    user_id_tensor = torch.tensor([user_id_mapping[user_id]], dtype=torch.long).to(device)
    game_ids = list(game_id_mapping.keys())
    recommendations = []
    with torch.no_grad():
        for game in game_ids:
            game_id = game_id_mapping[game]
            feature_vector = np.array(game_features[game]).astype(float)
            feature_tensor = torch.tensor([feature_vector], dtype=torch.float).to(device)
            output = model(feature_tensor)
            predicted_rating = output.item()
            recommendations.append((game, predicted_rating))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [game for game, _ in recommendations[:k]]

# 评估推荐效果
def evaluate_recommendations(recommended, actual_rated):
    recommended_set = set(recommended)
    actual_set = set(actual_rated)
    precision = len(recommended_set & actual_set) / len(recommended_set) if recommended else 0
    recall = len(recommended_set & actual_set) / len(actual_set) if actual_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1

# 主函数
def main():
    # 数据准备
    user_dict, game_features = load_data('user_game_data.csv')
    game_id_mapping = {game: idx for idx, game in enumerate(game_features.keys())}
    user_id_mapping = {user: idx for idx, user in enumerate(user_dict.keys())}
    
    # 数据集和数据加载器
    dataset = GameDataset(user_dict, game_features, game_id_mapping, user_id_mapping)
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 模型和训练
    input_dim = len(next(iter(game_features.values())))
    model = CNNRecommender(input_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, device, loader, optimizer, criterion)
    
    # 生成推荐
    target_user = 1
    recommended_games = generate_recommendations(model, device, target_user, game_features, game_id_mapping)
    print(f"推荐给用户{target_user}的游戏：{recommended_games}")
    
    # 评估
    actual_rated_games = [game for game, _ in user_dict[target_user]]
    precision, recall, f1 = evaluate_recommendations(recommended_games, actual_rated_games)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()

```
