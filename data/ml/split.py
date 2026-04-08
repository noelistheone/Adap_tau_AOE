import random
import os

# --- 配置 ---
INPUT_FILE = 'u.data'        # MovieLens 100k 的源数据文件
TRAIN_FILE = 'train.txt'     # 输出的训练集
TEST_FILE = 'test.txt'       # 输出的测试集
SPLIT_RATIO = 0.2            # 测试集占比 (0.2 = 20% = 1:4)
MIN_RATING = 0               # (可选) 过滤评分，例如设为3则只保留>=3分的记录。设为0保留所有。

def process_movielens_and_split():
    print(f"正在读取 {INPUT_FILE} ...")
    
    user_map = {}   # 旧 user_id -> 新 user_id
    item_map = {}   # 旧 item_id -> 新 item_id
    user_interactions = {} # {新uid: [新iid, ...]}

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t') # u.data 是 Tab 分隔的
                if len(parts) < 4:
                    continue
                
                # u.data 格式: user id | item id | rating | timestamp
                orig_uid = parts[0]
                orig_iid = parts[1]
                rating = int(parts[2])

                # 过滤低分交互 (如果需要)
                if rating < MIN_RATING:
                    continue

                # --- ID Remapping (映射为 0, 1, 2...) ---
                if orig_uid not in user_map:
                    user_map[orig_uid] = len(user_map)
                new_uid = user_map[orig_uid]

                if orig_iid not in item_map:
                    item_map[orig_iid] = len(item_map)
                new_iid = item_map[orig_iid]

                # --- 聚合 ---
                # REMAPPING_BUG: 
                # 这里我们是在读取一行的时候就进行 remapping。
                # 如果是 User Split，每个用户的 ID 都是独立的，最终 ID 是 0..N-1。因为我们对 user_interactions.keys() 排序后处理，user_id 能够保证是 0, 1, 2...
                # 但是如果是 Interaction Split，同一个 user id 可能在 train 和 test 都有，但 user id 必须保持一致且连续。

                # 如果我们需要的是 train.txt 里 User ID 是 0, 1, 2... 并且 Item ID 也是 0, 1, 2...
                # 那么只要在生成文件之前，确保所有的 item 和 user 都被映射过即可。
                # 这里的逻辑是：读取所有数据 -> 构建 map -> 生成 new_uid。这已经是正确的 remapping 逻辑了。
                # 为什么用户说不是从0开始呢？
                # 可能原因是：旧代码直接对 all_lines 进行了 shuffle，然后分成 train 和 test。
                # 比如总共 100 个用户，映射为 0..99。
                # Shuffle 后，train 拿到前 80 个用户，但这 80 个用户的 ID 可能是 [5, 99, 12, ...]。
                # 这就导致 train.txt 里的 user id 不是连续的 0..79。
                # 而推荐系统通常要求 ID 是连续整数以作为 Embedding 的索引。
                
                # 修复方案：
                # 1. 如果是 User Split (冷启动测试): 我们需要重新映射最终集合里的 ID。但通常这不是标准的 RS 评测方式。
                # 2. 如果是 Interaction Split (标准): 所有的 User 都在 Train 和 Test 中出现，ID 保持 0..N-1 即可。
                #    (上面的代码块我已经把它改成了 Interaction Split，这通常是用户想要的)
                #    这样 User ID 0..N-1 都会出现在 Train 和 Test 中 (只要交互足够多)。
                #    Item ID 也是 0..M-1。
                
                if new_uid not in user_interactions:
                    user_interactions[new_uid] = []
                user_interactions[new_uid].append(new_iid)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_FILE}。请确保已下载 ml-100k 数据集并解压。")
        return

    print(f"读取完成。共 {len(user_map)} 个用户, {len(item_map)} 个物品。")
    print("正在进行 Interaction Split (确保 User ID 连续 0..N)...")

    # 准备 Train 和 Test 容器
    train_lines = []
    test_lines = []
    
    random.seed(42)

    # 按照重新映射后的 user_id (0, 1, 2...) 顺序处理
    for uid in sorted(user_interactions.keys()):
        items = user_interactions[uid]
        if not items:
            continue
            
        # 随机打乱当前用户的交互物品
        random.shuffle(items)
        
        # 计算测试集数量
        n_test = int(len(items) * SPLIT_RATIO)
        
        # 策略：如果只有很少的数据，怎么分？
        # 这里采用保留至少一个在训练集的策略，当然具体取决于需求
        if len(items) > 1 and n_test == 0:
             n_test = 1 # 保证如果有多条数据，至少分一条去测试
        
        test_items = items[:n_test]
        train_items = items[n_test:]
        
        # 确保训练集不为空 (冷启动问题，如果不允许纯测试用户)
        if len(train_items) == 0: 
            # 如果全被分到测试集了（比如只有1条且n_test=1），则放回训练集
            train_items = test_items
            test_items = []

        # 转换为字符串并添加
        if train_items:
            train_items_str = [str(x) for x in train_items]
            train_lines.append(f"{uid} {' '.join(train_items_str)}")
        
        if test_items:
            test_items_str = [str(x) for x in test_items]
            test_lines.append(f"{uid} {' '.join(test_items_str)}")

    print(f"写入文件... (训练集用户数: {len(train_lines)}, 测试集用户数: {len(test_lines)})")

    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))

    print(f"处理完毕！生成文件: {TRAIN_FILE}, {TEST_FILE}")
    print("注意：User ID 和 Item ID 均已从 0 开始重新映射并保持连续。")

if __name__ == "__main__":
    process_movielens_and_split()
    train_data = all_lines[test_size:]

    print(f"写入文件... (训练集用户数: {len(train_data)}, 测试集用户数: {len(test_data)})")

    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data))
    
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_data))
    '''
    
    pass ## END Old logic switch

if __name__ == "__main__":
    process_movielens_and_split()