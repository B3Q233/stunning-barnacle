# 仓库侦察

## 检查目标

确认仓库根目录、版本、子模块、主要语言、目录职责、入口脚本、配置入口和源码调用关系。

## 优先位置

按顺序读取 `README*`、`pyproject.toml`、`setup.py`、`requirements*.txt`、`environment*.yml`、`Dockerfile`、`Makefile`、`scripts/`、`train*`、`test*`、`eval*`、`infer*` 和配置目录。用文本搜索定位 `main`、`argparse`、`hydra`、`yaml`、`load_state_dict`、`checkpoint` 等入口。

## 输出

记录仓库版本或 commit、顶层结构、入口文件、入口参数、关键调用链和每项结论的证据路径。不要只复制 README 的目录树。

## 只读验证

允许列目录、读取文本、查看 Git 状态/commit、搜索符号和查看脚本帮助。不要执行未知入口脚本，因为其可能下载数据、写缓存或启动训练。

## 常见误判

- README 的命令可能对应旧版本；优先核对当前入口和参数。
- 文件名包含 `train` 不等于可直接运行；必须追踪 `if __name__` 和参数解析。
- notebook、shell 包装器和软链接可能隐藏真实入口。