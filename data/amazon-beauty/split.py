import csv
import random
import os

# --- 配置 ---
INPUT_FILE = 'ratings_Beauty.csv'      # 你的源文件路径
TRAIN_FILE = 'train.txt'     # 输出的训练集文件名
TEST_FILE = 'test.txt'       # 输出的测试集文件名
SPLIT_RATIO = 0.2            # 测试集比例 (0.2 = 1:4)
HAS_HEADER = False           # 如果你的CSV第一行是标题(如 user_id,item_id)，请改为 True

def process_csv_and_split():
    print(f"正在读取 {INPUT_FILE} ...")
    
    # 临时存储原始数据 (str_uid -> [str_iid, ...])
    raw_user_interactions = {}

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            if HAS_HEADER:
                next(reader, None)

            for row in reader:
                if len(row) < 2: continue
                orig_uid, orig_iid = row[0].strip(), row[1].strip()
                if not orig_uid or not orig_iid: continue

                if orig_uid not in raw_user_interactions:
                    raw_user_interactions[orig_uid] = []
                raw_user_interactions[orig_uid].append(orig_iid)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    # --- 1. 过滤交互过少的用户 (可选，但推荐) ---
    # 只有交互数 >= 2 的用户才能既有 Train 又有 Test
    MIN_INTERACTIONS = 5 # 常用是 5 或 10，最少也得是 2
    print(f"过滤交互数少于 {MIN_INTERACTIONS} 的用户...")
    
    valid_users = [u for u, items in raw_user_interactions.items() if len(items) >= MIN_INTERACTIONS]
    
    # --- 2. 建立 ID 映射 (Remapping) ---
    # 仅对有效用户建立映射，确保最终 ID 0..N 都是有效的
    user_map = {u: i for i, u in enumerate(sorted(valid_users))}
    
    # 收集涉及到的所有 Item 并建立映射
    item_set = set()
    for u in valid_users:
        for item in raw_user_interactions[u]:
            item_set.add(item)
    item_map = {item: i for i, item in enumerate(sorted(list(item_set)))}

    print(f"过滤后剩余: 用户 {len(user_map)} 个, 物品 {len(item_map)} 个。")
    print("正在进行 Interaction Split...")

    train_lines = []
    test_lines = []
    
    random.seed(42)

    # 按新 ID 顺序处理 (0, 1, 2...)
    sorted_old_users = sorted(valid_users) # 对应 ID 0, 1, 2...
    
    for old_uid in sorted_old_users:
        new_uid = user_map[old_uid]
        # 获取该用户所有交互的 Item 新 ID
        items = [item_map[iid] for iid in raw_user_interactions[old_uid] if iid in item_map]
        
        # shuffle
        random.shuffle(items)
        
        # Split (保证 Train 和 Test 至少各有一个，除非总数不够)
        n_total = len(items)
        n_test = int(n_total * SPLIT_RATIO)
        if n_test < 1: n_test = 1 # 至少分 1 个给测试集
        
        # 此时因为我们要保证 Train 也不为空 (用户交互 >= 2 才能保证)
        # 如果只有 1 个交互，上面过滤 >=2 已经排除了
        
        test_items = items[:n_test]
        train_items = items[n_test:]
        
        # 双重保险
        if not train_items: # 这种情况应该被 MIN_INTERACTIONS=2 避免
             train_items = test_items
             test_items = [] # 牺牲测试集保训练集

        # 写入
        if train_items:
            # 格式: new_uid item1 item2 ...
            train_lines.append(f"{new_uid} {' '.join(map(str, train_items))}")
        
        # 如果某些评测代码要求 Test 文件里不仅仅是包含交互的用户，而是所有用户
        # 那么即使用户没有 Test Item，通常也不会在 test.txt 写一行只有 uid 的空行
        # 这里只写入有测试集的用户
        if test_items:
            test_lines.append(f"{new_uid} {' '.join(map(str, test_items))}")


    # 4. 写入文件
    print(f"正在写入文件 (训练集: {len(train_lines)}行, 测试集: {len(test_lines)}行)...")

    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))

    # 5. 保存映射表 (可选，方便后续查阅)
    # with open('user_map.txt', 'w') as f:
    #    for k, v in user_map.items(): f.write(f"{k} {v}\n")
    
    print("处理完毕！")
    print(f"生成文件: {TRAIN_FILE}, {TEST_FILE}")
    print("注意：User ID 和 Item ID 均已从 0 开始重新映射并保持连续。")

if __name__ == "__main__":
    process_csv_and_split()