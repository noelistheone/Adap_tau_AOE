import os

def process_data(train_input, test_input, train_output, test_output):
    # 1. 初始化全局映射字典
    user_map = {}  # 旧 user_id -> 新 user_id
    item_map = {}  # 旧 item_id -> 新 item_id
    
    # 用于存储聚合结果
    train_interactions = {}
    test_interactions = {}

    print("正在第一遍扫描 (建立 ID 映射)...")

    # 定义一个辅助函数来读取文件并构建 Map
    def read_and_map(filename, target_interactions):
        if not os.path.exists(filename):
            print(f"警告: 文件 {filename} 不存在，跳过。")
            return

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                # 兼容不同格式: user item rating 或 user item
                if len(parts) < 2:
                    continue
                
                orig_uid = parts[0]
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
                if new_uid not in target_interactions:
                    target_interactions[new_uid] = []
                target_interactions[new_uid].append(new_iid)

    # 依次读取 train 和 test，共享 map
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
# 之前的配置里写的是 test_.txt -> test.txt，所以我推测原始文件带下划线
train_in = 'train_.txt' 
test_in = 'test_.txt'
train_out = 'train.txt'
test_out = 'test.txt'

# 运行处理函数
if __name__ == "__main__":
    process_data(train_in, test_in, train_out, test_out)