"""WMF 配置键名常量（唯一定义来源）

所有读取 config 的地方必须从本模块导入键名，禁止在多个文件里各自
手写字面字符串，避免"同一概念两个拼写"的隐性 bug。
"""

# data 模块
KEY_DATASET = "dataset"
KEY_RAW_DATA_PATH = "raw_data_path"
KEY_PROCESSED_DATA_PATH = "processed_data_path"
KEY_VAL_RATIO = "val_ratio"

# 数据/模型元信息
KEY_NUM_USERS = "num_users"
KEY_NUM_ITEMS = "num_items"

# model 模块
KEY_FACTORS = "factors"
KEY_ALPHA = "alpha"
KEY_EPSILON = "epsilon"
KEY_CONFIDENCE_SCHEME = "confidence_scheme"
KEY_INIT_METHOD = "init_method"
KEY_INIT_STD = "init_std"

# training 模块
KEY_LAMBDA_REG = "lambda_reg"
KEY_EPOCHS = "epochs"
KEY_BATCH_SIZE = "batch_size"
KEY_DEVICE = "device"
KEY_SAVE_EVERY_N_EPOCHS = "save_every_n_epochs"
KEY_SHUFFLE = "shuffle"
KEY_RUN_TAG = "run_tag"

# evaluation 模块
KEY_K = "k"
KEY_EVAL_EVERY = "eval_every"
KEY_METRICS = "metrics"
KEY_CHECKPOINT_MODE = "checkpoint_mode"
