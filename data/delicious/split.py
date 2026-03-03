import os
import random

def process_and_split_data(input_file, train_file, test_file, split_ratio=0.8):
    print(f"1. 正在读取文件: {input_file} ...")
    
    # 设定随机种子，保证每次运行结果一致
    random.seed(2024)
    
    # 映射字典
    user_map = {}
    item_map = {}
    
    # 存储聚合后的数据: {new_user_id: [new_item_id1, new_item_id2, ...]}
    # 使用 set 去重，防止同一个用户对同一个 item 多次打标签造成重复
    user_interactions = {}

    # --- 第一步：读取并重映射 ID ---
    with open(input_file, 'r', encoding='utf-8') as f:
        # 跳过第一行表头
        header = next(f)
        
        for line in f:
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            
            # user_taggedbookmarks.dat 格式: userID \t bookmarkID \t ...
            raw_user_id = parts[0]
            raw_item_id = parts[1]
            
            # ID Remapping: 遇到新ID就分配一个从0开始的整数
            if raw_user_id not in user_map:
                user_map[raw_user_id] = len(user_map)
            if raw_item_id not in item_map:
                item_map[raw_item_id] = len(item_map)
                
            u_id = user_map[raw_user_id]
            i_id = item_map[raw_item_id]
            
            if u_id not in user_interactions:
                user_interactions[u_id] = set()
            user_interactions[u_id].add(i_id)

    print(f"   统计: 共有 {len(user_map)} 个用户, {len(item_map)} 个物品。")

    # --- 第二步：切分 Train/Test 并写入文件 ---
    print(f"2. 正在进行 8:2 切分并写入文件...")
    
    train_count = 0
    test_count = 0

    with open(train_file, 'w', encoding='utf-8') as f_train, \
         open(test_file, 'w', encoding='utf-8') as f_test:
        
        # 按 UserID 顺序处理
        for u_id in sorted(user_interactions.keys()):
            items = list(user_interactions[u_id])
            
            # 随机打乱该用户的交互列表
            random.shuffle(items)
            
            # 计算切分点
            n_items = len(items)
            n_train = int(n_items * split_ratio)
            
            # 确保如果用户交互太少（例如只有1个），至少分到训练集，避免冷启动问题过于严重
            # 如果你想严格执行比例（哪怕测试集为空），可以去掉下面这个 if
            if n_train == 0 and n_items > 0:
                n_train = 1
            
            train_items = items[:n_train]
            test_items = items[n_train:]
            
            # 写入 Train
            if train_items:
                line = f"{u_id} {' '.join(map(str, train_items))}\n"
                f_train.write(line)
                train_count += len(train_items)
                
            # 写入 Test
            if test_items:
                line = f"{u_id} {' '.join(map(str, test_items))}\n"
                f_test.write(line)
                test_count += len(test_items)

    print(f"处理完成！")
    print(f"生成的训练集: {train_file} (包含 {train_count} 次交互)")
    print(f"生成的测试集: {test_file} (包含 {test_count} 次交互)")

# --- 配置与运行 ---
input_path = 'user_taggedbookmarks.dat'  # 原始文件名
train_path = 'train.txt'                 # 输出的训练集
test_path = 'test.txt'                   # 输出的测试集

if __name__ == '__main__':
    if os.path.exists(input_path):
        process_and_split_data(input_path, train_path, test_path)
    else:
        print(f"错误: 找不到文件 {input_path}")