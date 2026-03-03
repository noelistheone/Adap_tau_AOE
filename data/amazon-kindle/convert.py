import os
from collections import defaultdict

def process_data(train_input, test_input, train_output, test_output):
    # 1. 统计每个用户的总交互数
    user_interaction_counts = defaultdict(int)
    
    def count_interactions(filename):
        if not os.path.exists(filename):
            return
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                orig_uid = parts[0]
                user_interaction_counts[orig_uid] += 1

    print("正在统计用户交互数量...")
    count_interactions(train_input)
    count_interactions(test_input)
    
    # 2. 选出交互数量位于前20%的用户
    total_users = len(user_interaction_counts)
    top_20_percent_count = max(1, int(total_users * 0.2))
    
    # 按交互数降序排序
    sorted_users = sorted(user_interaction_counts.items(), key=lambda x: x[1], reverse=True)
    top_users = set([uid for uid, count in sorted_users[:top_20_percent_count]])
    
    print(f"总用户数: {total_users}, 选出前20%用户数: {len(top_users)}")

    # 3. 重新映射ID并提取数据
    user_map = {}  # 旧 user_id -> 新 user_id
    item_map = {}  # 旧 item_id -> 新 item_id
    
    train_interactions = defaultdict(list)
    test_interactions = defaultdict(list)

    print("正在建立 ID 映射并提取前20%用户数据...")

    def read_and_map(filename, target_interactions):
        if not os.path.exists(filename):
            print(f"警告: 文件 {filename} 不存在，跳过。")
            return

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                orig_uid = parts[0]
                if orig_uid not in top_users:
                    continue  # 过滤掉非前20%的用户
                    
                orig_iid = parts[1]
                
                # --- User ID Remap (Global) ---
                if orig_uid not in user_map:
                    user_map[orig_uid] = len(user_map)
                new_uid = user_map[orig_uid]
                
                # --- Item ID Remap (Global) ---
                if orig_iid not in item_map:
                    item_map[orig_iid] = len(item_map)
                new_iid = item_map[orig_iid]
                
                # --- 存储 ---
                target_interactions[new_uid].append(new_iid)

    print(f"读取 Train: {train_input}")
    read_and_map(train_input, train_interactions)
    
    print(f"读取 Test: {test_input}")
    read_and_map(test_input, test_interactions)

    print(f"映射构建完成。共 {len(user_map)} 用户，{len(item_map)} 物品。")
    print("正在写入输出文件...")

    # 定义写入函数
    def write_output(filename, interactions):
        with open(filename, 'w', encoding='utf-8') as f_out:
            # 按 user_id 排序写入
            for uid in sorted(interactions.keys()):
                items = interactions[uid]
                items_str = [str(iid) for iid in items]
                line = f"{uid} {' '.join(items_str)}\n"
                f_out.write(line)
        print(f"写入成功: {filename}")

    write_output(train_output, train_interactions)
    write_output(test_output, test_interactions)

# --- 配置部分 ---
# 假设原始文件是 train_.txt 和 test_.txt (如果不确定文件名，请检查你的目录)
# 之前的配置里写的是 train_.txt -> train.txt
train_in = 'train_.txt' 
test_in = 'test_.txt'
train_out = 'train.txt'
test_out = 'test.txt'

# 运行处理函数
if __name__ == "__main__":
    process_data(train_in, test_in, train_out, test_out)