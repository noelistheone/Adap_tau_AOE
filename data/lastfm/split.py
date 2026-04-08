import os
import random

def process_lastfm_data(input_file, train_file, test_file, split_ratio=0.8):
    print(f"1. 正在读取文件: {input_file} ...")
    
    # 设定随机种子，确保结果可复现
    random.seed(2024)
    
    # 映射字典 {原始ID: 新ID}
    user_map = {}
    item_map = {}
    
    # 存储交互: {new_user_id: [new_item_id, ...]}
    # 使用 set 自动去重（防止文件中有重复行）
    user_interactions = {}

    # --- 第一步：读取并映射 ID ---
    with open(input_file, 'r', encoding='utf-8') as f:
        # 跳过第一行表头 (userID	artistID	weight)
        header = next(f)
        
        for line in f:
            if not line.strip():
                continue
            
            # 分割数据 (Tab 分隔)
            parts = line.strip().split('\t')
            
            # 获取原始 ID
            raw_user_id = parts[0]
            raw_item_id = parts[1]
            # parts[2] 是 weight，但在隐式反馈 Top-K 推荐中通常忽略具体数值，只看有无交互
            
            # ID Remapping
            if raw_user_id not in user_map:
                user_map[raw_user_id] = len(user_map)
            if raw_item_id not in item_map:
                item_map[raw_item_id] = len(item_map)
                
            u_id = user_map[raw_user_id]
            i_id = item_map[raw_item_id]
            
            if u_id not in user_interactions:
                user_interactions[u_id] = set()
            
            user_interactions[u_id].add(i_id)

    print(f"   统计: 共有 {len(user_map)} 个用户, {len(item_map)} 个歌手(Artists)。")

    # --- 第二步：切分 Train/Test 并写入 ---
    print(f"2. 正在进行按用户 8:2 切分...")
    
    train_count = 0
    test_count = 0

    with open(train_file, 'w', encoding='utf-8') as f_train, \
         open(test_file, 'w', encoding='utf-8') as f_test:
        
        # 按照新的 UserID 顺序处理 (0, 1, 2...)
        for u_id in sorted(user_interactions.keys()):
            items = list(user_interactions[u_id])
            
            # 随机打乱
            random.shuffle(items)
            
            # 计算切分点
            n_items = len(items)
            n_train = int(n_items * split_ratio)
            
            # 保护机制：如果用户交互很少（例如只有1个），强制放入训练集
            # 这样保证每个用户至少在 Embedding 层能被学习到
            if n_train == 0 and n_items > 0:
                n_train = 1
            
            train_items = items[:n_train]
            test_items = items[n_train:]
            
            # 写入 Train (格式: UserID ItemID1 ItemID2 ...)
            if train_items:
                line = f"{u_id} {' '.join(map(str, train_items))}\n"
                f_train.write(line)
                train_count += len(train_items)
                
            # 写入 Test
            if test_items:
                line = f"{u_id} {' '.join(map(str, test_items))}\n"
                f_test.write(line)
                test_count += len(test_items)

    print("-" * 30)
    print(f"处理完成！")
    print(f"训练集: {train_file} (交互数: {train_count})")
    print(f"测试集: {test_file} (交互数: {test_count})")
    print(f"总交互数: {train_count + test_count}")

# --- 配置 ---
input_path = 'user_artists.dat'
output_train = 'train.txt'
output_test = 'test.txt'

if __name__ == '__main__':
    if os.path.exists(input_path):
        process_lastfm_data(input_path, output_train, output_test)
    else:
        print(f"错误: 找不到文件 {input_path}，请确保文件在当前目录下。")